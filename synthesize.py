"""
Синтез фазового ДОЭ по 4 изображениям — формат сдачи из ТЗ (слайд 6).

Вход:  4 изображения 128x128, 8 бит, одноканальные png. Порядок = расстояния 24.1 / 25.3 / 26.5 / 27.7 мм.
Выход (в папке --out):
  doe.png                 фаза ДОЭ, 8 бит: 0 -> -pi, 255 -> +pi
  recon_1..4_z<мм>.png    восстановления по формуле Френеля из 8-битного ДОЭ (амплитуда, нормировка на max, 8 бит)
  metrics.json / .csv     КК (Пирсон) и SSIM для каждой пары «исходное — восстановленное», средние, время синтеза

Методы (--method):
  m10p     нейросетевой: ядро «GS + веса + связь» x20 + 3 обученных шага по метрикам, обучен на смеси (runs/f1_m10p_mix)   [по умолчанию]
  m10      нейросетевой: GS x20 + 3 обученных шага по метрикам, обучен на срезах (runs/s1_m10)
  m2       нейросеть за один проход (runs/e11_m2_long)
  hybrid   без сети: GS x20 -> 10 шагов Adam по метрикам
  gs       классический многоплоскостной Герчберг-Сэкстон (--iters, по умолчанию 300)
  gsplus   классика с весами плоскостей и попиксельной обратной связью (Э15)

Пример:
  python3 -I synthesize.py a.png b.png c.png d.png --out result/
  python3 -I synthesize.py --dir папка_с_4_png --method gs --iters 300 --out result_gs/
"""
import os, sys, json, csv, time, argparse, glob
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from PIL import Image
from physics import Fresnel, evaluate, quantize_phase, loss_fn, gs, gs_plus, Z, N, pick_device

DIST_MM = [z * 1e3 for z in Z]


def load_inputs(paths):
    imgs = []
    for p in paths:
        im = Image.open(p)
        if im.mode != "L":
            print(f"предупреждение: {p} в режиме {im.mode}, перевожу в L (grayscale)")
            im = im.convert("L")
        if im.size != (N, N):
            raise SystemExit(f"{p}: размер {im.size}, требуется {N}x{N}")
        imgs.append(np.asarray(im).astype(np.float32) / 255)
    T = torch.from_numpy(np.stack(imgs))[None]                      # (1,4,N,N)
    return T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)       # нормировка на max = 1, как у жюри


def build(method, iters, dev):
    if method == "gs":
        return lambda T, f: gs(T, f, iters=iters)
    if method == "gsplus":
        return lambda T, f: gs_plus(T, f, iters=iters)
    if method == "hybrid":
        def run(T, f, g=20, k=10, lr=0.1):
            with torch.no_grad():
                phi = gs(T, f, iters=g)
            p = phi.clone().requires_grad_(True)
            opt = torch.optim.Adam([p], lr=lr)
            for _ in range(k):
                opt.zero_grad(); l, _, _ = loss_fn(p, T, f); l.backward(); opt.step()
            return p.detach()
        return run
    from models import MODELS
    ckpt = {"m10": "s1_m10", "m10p": "f1_m10p_mix", "m2": "e11_m2_long"}[method]
    m = MODELS[method]().to(dev)
    m.load_state_dict(torch.load(os.path.join(ROOT, "runs", ckpt, "model.pt"), map_location=dev))
    m.eval()

    def run(T, f):
        with torch.no_grad():
            return m(T, f).detach()
    return run


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="4 png в порядке расстояний")
    ap.add_argument("--dir", help="папка с 4 png (берутся в алфавитном порядке)")
    ap.add_argument("--out", default="result")
    ap.add_argument("--method", default="m10p", choices=["m10p", "m10", "m2", "hybrid", "gs", "gsplus"])
    ap.add_argument("--iters", type=int, default=300, help="итерации для --method gs / gsplus")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    paths = args.images or sorted(glob.glob(os.path.join(args.dir, "*.png")))
    if len(paths) != 4:
        raise SystemExit(f"нужно ровно 4 png, получено {len(paths)}")
    dev = args.device or pick_device()
    f = Fresnel().to(dev)
    T = load_inputs(paths).to(dev)
    run = build(args.method, args.iters, dev)

    run(T, f); sync(dev)                                             # прогрев (не входит во время)
    t0 = time.perf_counter(); phi = run(T, f); sync(dev); dt = time.perf_counter() - t0

    k, _ = quantize_phase(phi)
    p, s, rec = evaluate(phi, T, f)                                 # восстановление из 8-битного ДОЭ, как у жюри
    os.makedirs(args.out, exist_ok=True)
    Image.fromarray(k[0].cpu().numpy().astype(np.uint8), mode="L").save(os.path.join(args.out, "doe.png"))
    rows = []
    for j in range(4):
        name = f"recon_{j + 1}_z{DIST_MM[j]:.1f}mm.png"
        Image.fromarray((rec[0, j] * 255).round().cpu().numpy().astype(np.uint8), mode="L").save(os.path.join(args.out, name))
        rows.append(dict(plane=j + 1, distance_mm=round(DIST_MM[j], 1), source=os.path.basename(paths[j]), recon=name,
                         pearson=round(p[0, j].item(), 4), ssim=round(s[0, j].item(), 4)))
    summary = dict(method=args.method, device=dev, synthesis_time_ms=round(dt * 1e3, 2),
                   pearson_mean=round(p.mean().item(), 4), ssim_mean=round(s.mean().item(), 4),
                   pearson_min=round(p.min().item(), 4), ssim_min=round(s.min().item(), 4), planes=rows)
    with open(os.path.join(args.out, "metrics.json"), "w") as fo:
        json.dump(summary, fo, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "metrics.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print(f"метод {args.method}, устройство {dev}, время синтеза ДОЭ {dt * 1e3:.1f} мс")
    for r in rows:
        print(f"  плоскость {r['plane']} ({r['distance_mm']} мм): КК {r['pearson']:.3f}  SSIM {r['ssim']:.3f}  <- {r['source']}")
    print(f"  среднее: КК {summary['pearson_mean']:.3f}  SSIM {summary['ssim_mean']:.3f}")
    print(f"сохранено в {args.out}/: doe.png, recon_*.png, metrics.json, metrics.csv")


if __name__ == "__main__":
    main()
