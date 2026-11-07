"""
Генерация обучающих четвёрок-слоёв для синтеза многоплоскостного ДОЭ.

Идея (research_log.md, В8): тестовые 4 изображения — слои одной 3D-сцены (допущение команды).
Поэтому каждая обучающая четвёрка = одно фото, разрезанное по карте глубины на 4 слоя.

Шаг 1 (один раз): пул сцен
    python3 -I data.py prepare data/raw/coco/val2017 data/pool_coco.npz
    python3 -I data.py prepare data/raw/div2k/DIV2K_valid_HR data/pool_div2k.npz
  Для каждой картинки: центральный квадрат → grayscale 160x160 (uint8)
  + карта относительной глубины от Depth Anything V2 Small (uint8, БОЛЬШЕ = БЛИЖЕ).

Шаг 2: четвёрки режутся на лету
    from data import LayerDataset
    ds = LayerDataset("data/pool_coco.npz")      # ds[i] -> float32 тензор (4,128,128) в [0,1]

Шаг 3 (проверка): сетка примеров и статистика слоёв
    python3 -I data.py preview data/pool_coco.npz out/layers_preview.png

Режимы LayerDataset.mode:
  "slice"        — D-L1: срезы одной сцены по глубине на тёмном фоне (основной)
  "independent"  — D-A: 4 независимых фото (контрольный эксперимент)
Глубина: depth_source="model" (из пула) или "synthetic" (случайная гладкая, без модели).
Порядок слоёв: слой 0 = ближний = 24.1 мм (допущение).
"""
import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

POOL_SIZE = 160      # сторона хранимого квадрата
OUT = 128            # сторона слоя
N_LAYERS = 4
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"   # лицензия Apache-2.0


# ---------------------------------------------------------------- пул сцен
def _center_square(im):
    w, h = im.size
    s = min(w, h)
    l, t = (w - s) // 2, (h - s) // 2
    return im.crop((l, t, l + s, t + s))


def prepare(src_dir, out_path, limit=None, batch=16):
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    files = sorted(glob.glob(os.path.join(src_dir, "*.jpg")) + glob.glob(os.path.join(src_dir, "*.png")))
    if limit:
        files = files[: int(limit)]
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    proc = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
    model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL).to(dev).eval()

    imgs, depths, names = [], [], []
    t0 = time.time()
    for b in range(0, len(files), batch):
        crops = []
        for f in files[b: b + batch]:
            try:
                im = _center_square(Image.open(f).convert("RGB"))
            except Exception as e:                       # битый файл — пропускаем
                print("skip", f, e)
                continue
            crops.append(im)
            names.append(os.path.basename(f))
            imgs.append(np.asarray(im.convert("L").resize((POOL_SIZE, POOL_SIZE), Image.BICUBIC)))
        if not crops:
            continue
        with torch.no_grad():
            inp = proc(images=crops, return_tensors="pt").to(dev)
            d = model(**inp).predicted_depth                      # (B,H,W), относительная обратная глубина
            d = F.interpolate(d[:, None].float(), size=(POOL_SIZE, POOL_SIZE), mode="bilinear", align_corners=False)[:, 0]
        d = d.cpu().numpy()
        for di in d:
            lo, hi = np.percentile(di, 0.5), np.percentile(di, 99.5)
            depths.append((np.clip((di - lo) / (hi - lo + 1e-8), 0, 1) * 255).round().astype(np.uint8))
        done = b + len(crops)
        if (b // batch) % 20 == 0:
            el = time.time() - t0
            print(f"{done}/{len(files)}  {el:.0f} c  (~{el / done * (len(files) - done):.0f} c осталось)", flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(out_path, img=np.stack(imgs), depth=np.stack(depths), names=np.array(names))
    print(f"сохранено {len(imgs)} сцен -> {out_path}  ({os.path.getsize(out_path) / 1e6:.0f} МБ, {time.time() - t0:.0f} c)")


def load_pool(path):
    z = np.load(path)
    return z["img"], z["depth"]


# ---------------------------------------------------------------- генерация четвёрок
def synthetic_depth(rng, size=POOL_SIZE):
    """случайная гладкая «глубина»: наклонная плоскость + гауссовы холмы (больше = ближе)"""
    yy, xx = np.mgrid[0:size, 0:size] / size
    a = rng.uniform(-1, 1, 2)
    d = a[0] * xx + a[1] * yy
    for _ in range(rng.integers(2, 7)):
        c = rng.uniform(0, 1, 2); s = rng.uniform(0.05, 0.3); h = rng.uniform(0.5, 2.0)
        d = d + h * np.exp(-((xx - c[0]) ** 2 + (yy - c[1]) ** 2) / (2 * s ** 2))
    d = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return (d * 255).astype(np.uint8)


class LayerDataset(torch.utils.data.Dataset):
    def __init__(self, pool_path, length=10000, mode="slice", depth_source="model",
                 aug=True, min_frac=0.05, seed=None):
        self.img, self.depth = load_pool(pool_path)
        self.length, self.mode, self.depth_source, self.aug, self.min_frac = length, mode, depth_source, aug, min_frac
        self.seed = seed

    def __len__(self):
        return self.length

    def _rng(self, i):
        # детерминированно при заданном seed (валидация), иначе случайно (обучение)
        if self.seed is None:
            return np.random.default_rng()
        return np.random.default_rng([self.seed, i])

    def _crop(self, rng, img, depth):
        c = int(rng.integers(OUT, POOL_SIZE + 1))                   # масштабная аугментация
        y, x = rng.integers(0, POOL_SIZE - c + 1, 2)
        im = torch.from_numpy(img[y:y + c, x:x + c].astype(np.float32) / 255)[None, None]
        de = torch.from_numpy(depth[y:y + c, x:x + c].astype(np.float32) / 255)[None, None]
        if c != OUT:
            im = F.interpolate(im, size=(OUT, OUT), mode="bilinear", align_corners=False, antialias=True)
            de = F.interpolate(de, size=(OUT, OUT), mode="bilinear", align_corners=False)
        im, de = im[0, 0], de[0, 0]
        if self.aug and rng.random() < 0.5:
            im, de = im.flip(-1), de.flip(-1)
        return im, de

    def _photometric(self, rng, im):
        if not self.aug:
            return im
        im = im.clamp(0, 1) ** float(rng.uniform(0.7, 1.4))                       # гамма
        m = im.mean()
        im = (im - m) * float(rng.uniform(0.8, 1.25)) + m + float(rng.uniform(-0.1, 0.1))  # контраст, яркость
        return im.clamp(0, 1)

    def _slice(self, rng, im, de):
        """делим глубину на 4 интервала: базовые квантили 25/50/75% + случайный сдвиг"""
        q = np.sort(np.clip(np.array([0.25, 0.5, 0.75]) + rng.uniform(-0.1, 0.1, 3), 0.05, 0.95))
        t = torch.quantile(de.flatten(), torch.tensor(q, dtype=torch.float32))
        # слой 0 = самые большие значения = ближний план
        idx = (N_LAYERS - 1) - torch.bucketize(de, t)                 # 0..3
        layers = torch.stack([im * (idx == k) for k in range(N_LAYERS)])
        return layers

    def _finalize(self, layers):
        # 8 бит, как у реальных png, и нормировка на max = 1 (как у жюри)
        layers = torch.round(layers * 255) / 255
        mx = layers.amax((-2, -1), keepdim=True)
        return layers / mx.clamp_min(1e-8)

    def _ok(self, layers):
        mx = layers.amax((-2, -1))
        frac = (layers > 1 / 255).float().mean((-2, -1))
        return bool((mx > 0.1).all() and (frac > self.min_frac).all())

    def _objects(self, rng):
        """«объекты на чёрном фоне»: в каждом слое 1–3 фигуры (эллипсы, повёрнутые прямоугольники),
        залитые кропом своего фото. Похоже на объекты на разных глубинах, как кольца на слайде 2 ТЗ."""
        yy, xx = np.mgrid[0:OUT, 0:OUT].astype(np.float32)
        layers = []
        for _ in range(N_LAYERS):
            k = int(rng.integers(0, len(self.img)))
            im = self._photometric(rng, self._crop(rng, self.img[k], self.depth[k])[0]).numpy()
            m = np.zeros((OUT, OUT), bool)
            for _ in range(int(rng.integers(1, 4))):
                cx, cy = rng.uniform(16, OUT - 16, 2)
                a, b = rng.uniform(8, 40, 2)
                t = rng.uniform(0, np.pi)
                u = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
                v = -(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)
                m |= ((u / a) ** 2 + (v / b) ** 2 < 1) if rng.random() < 0.5 else ((np.abs(u) < a) & (np.abs(v) < b))
            layers.append(torch.from_numpy(im * m))
        return torch.stack(layers)

    def __getitem__(self, i):
        rng = self._rng(i)
        for _ in range(20):                                           # пересэмплируем пустые/тёмные слои
            mode = self.mode
            if mode == "mix":                                         # Г15: срезы 50% / независимые 30% / объекты 20%
                mode = rng.choice(["slice", "independent", "objects"], p=[0.5, 0.3, 0.2])
            if mode == "objects":
                layers = self._objects(rng)
            elif mode == "independent":
                ks = rng.integers(0, len(self.img), N_LAYERS)
                layers = torch.stack([self._photometric(rng, self._crop(rng, self.img[k], self.depth[k])[0]) for k in ks])
            else:
                k = int(rng.integers(0, len(self.img)))
                d = self.depth[k] if self.depth_source == "model" else synthetic_depth(rng)
                im, de = self._crop(rng, self.img[k], d)
                layers = self._slice(rng, self._photometric(rng, im), de)
            if self._ok(layers):
                return self._finalize(layers)
        return self._finalize(layers)


# ---------------------------------------------------------------- проверка
def preview(pool_path, out_png, n=6, n_stats=1000):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img, depth = load_pool(pool_path)
    ds = LayerDataset(pool_path, mode="slice", seed=0)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, len(img), n)
    fig, ax = plt.subplots(n, 6, figsize=(13, 2.3 * n))
    for r, k in enumerate(ids):
        ax[r, 0].imshow(img[k], cmap="gray"); ax[r, 0].set_title("фото", fontsize=8)
        ax[r, 1].imshow(depth[k], cmap="magma"); ax[r, 1].set_title("глубина (светлее = ближе)", fontsize=8)
        d = ds._crop(np.random.default_rng(int(k)), img[k], depth[k])
        L = ds._finalize(ds._slice(np.random.default_rng(int(k)), d[0], d[1]))
        for j in range(4):
            ax[r, 2 + j].imshow(L[j], cmap="gray", vmin=0, vmax=1)
            ax[r, 2 + j].set_title(f"слой {j} ({['24.1', '25.3', '26.5', '27.7'][j]} мм)", fontsize=8)
    for a in ax.flat:
        a.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=100)

    # статистика по случайным четвёркам (с аугментациями, как при обучении)
    ds_tr = LayerDataset(pool_path, mode="slice", seed=1)
    S = torch.stack([ds_tr[i] for i in range(n_stats)])          # (n,4,128,128)
    nz = (S > 1 / 255).float()
    frac = nz.mean((-2, -1))
    mean_nz = (S * nz).sum((-2, -1)) / nz.sum((-2, -1)).clamp_min(1)
    print(f"пул: {len(img)} сцен, {pool_path}")
    print("доля непустых пикселей по слоям :", " ".join(f"{v:.2f}" for v in frac.mean(0).tolist()),
          f"(мин {frac.min():.2f}, макс {frac.max():.2f})")
    print("средняя яркость непустых пикселей:", " ".join(f"{v:.2f}" for v in mean_nz.mean(0).tolist()))
    print(f"сохранено: {out_png}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        prepare(sys.argv[2], sys.argv[3], limit=(sys.argv[4] if len(sys.argv) > 4 else None))
    elif cmd == "preview":
        preview(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
