"""
Бейзлайны синтеза многоплоскостного фазового ДОЭ (4 изображения -> 1 фаза 128x128).

Физика ровно по ТЗ: прямой расчёт дифракции Френеля с одним БПФ
    K_k(x,y) ~ chirp_out * FFT( exp(i*phi(u,v)) * exp(i*pi/(lam*z_k) * (u^2+v^2)) )
Интенсивность |K_k|^2 от внешнего множителя (exp(jkz)/(j*lam*z) * chirp_out) не зависит.

Методы:
  superposition  - один проход: сумма обратно распространённых полей, берём фазу
  gs             - многоплоскостной (параллельный) Герчберг-Сэкстон
  adam           - прямая градиентная оптимизация фазы под метрики (PCC + SSIM)

Запуск:  python3 doe_baseline.py            (картинки генерятся сами)
         python3 doe_baseline.py a.png b.png c.png d.png
"""
import sys, time, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

torch.manual_seed(0)
DEV = "cpu"  # 128x128 БПФ слишком мелкие, на CPU быстрее чем на MPS

# ---------------- параметры из ТЗ ----------------
N = 128
DX = 10e-6                       # пиксель ПВМС, м
LAM = 532e-9                     # длина волны, м
Z = [24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3]

def report_geometry():
    zc = N * DX**2 / LAM
    print(f"критическое расстояние z_c = N*dx^2/lam = {zc*1e3:.2f} мм")
    for z in Z:
        print(f"  z={z*1e3:.1f} мм -> пиксель в плоскости восстановления {LAM*z/(N*DX)*1e6:.2f} мкм")
    na = (N * DX / 2) / Z[0]
    print(f"NA ~ {na:.4f}, глубина резкости ~ lam/NA^2 = {LAM/na**2*1e3:.2f} мм, шаг между плоскостями 1.2 мм")

# ---------------- оператор Френеля ----------------
c = (torch.arange(N, dtype=torch.float64) - N / 2) * DX
UU, VV = torch.meshgrid(c, c, indexing="ij")
R2 = (UU**2 + VV**2)
CHIRPS = torch.stack([torch.exp(1j * torch.pi / (LAM * z) * R2) for z in Z]).to(torch.complex64).to(DEV)  # (4,N,N)

def fresnel(phi):
    """phi: (N,N) или (B,N,N) -> интенсивности (B,4,N,N)"""
    if phi.dim() == 2:
        phi = phi[None]
    field = torch.exp(1j * phi)[:, None] * CHIRPS[None]          # (B,4,N,N)
    out = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1)) / N
    return out.abs() ** 2

def fresnel_field(phi):
    field = torch.exp(1j * phi)[None] * CHIRPS
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field, dim=(-2, -1))), dim=(-2, -1)) / N

def back(fields):
    """поля в 4 плоскостях (4,N,N) -> поля в плоскости ДОЭ (4,N,N)"""
    h = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(fields, dim=(-2, -1))), dim=(-2, -1)) * N
    return h * CHIRPS.conj()

# ---------------- метрики ----------------
def pcc(a, b):
    a = a - a.mean(dim=(-2, -1), keepdim=True)
    b = b - b.mean(dim=(-2, -1), keepdim=True)
    return (a * b).sum((-2, -1)) / torch.sqrt((a**2).sum((-2, -1)) * (b**2).sum((-2, -1)) + 1e-12)

def _gauss(win=11, sigma=1.5):
    g = torch.exp(-((torch.arange(win) - win // 2) ** 2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None]).float()

GW = _gauss().to(DEV)[None, None]

def ssim(a, b):
    """a,b: (...,N,N) в [0,1]; классический SSIM, окно Гаусса 11x11"""
    shp = a.shape
    a = a.reshape(-1, 1, N, N); b = b.reshape(-1, 1, N, N)
    c1, c2 = 0.01**2, 0.03**2
    mu_a = F.conv2d(a, GW); mu_b = F.conv2d(b, GW)
    saa = F.conv2d(a * a, GW) - mu_a**2
    sbb = F.conv2d(b * b, GW) - mu_b**2
    sab = F.conv2d(a * b, GW) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (saa + sbb + c2))
    return m.mean((-2, -1)).reshape(shp[:-2])

def fit_scale(I, T):
    """оптимальный масштаб интенсивности под цель (МНК), затем клип в [0,1]"""
    s = (I * T).sum((-2, -1), keepdim=True) / ((I * I).sum((-2, -1), keepdim=True) + 1e-12)
    return (I * s).clamp(0, 1)

def quantize_phase(phi):
    """как в ТЗ: 0 -> -pi, 255 -> +pi"""
    w = torch.remainder(phi + torch.pi, 2 * torch.pi)  # [0, 2pi)
    k = torch.round(w / (2 * torch.pi) * 255).clamp(0, 255)
    return k, k / 255 * 2 * torch.pi - torch.pi

def to_image(I):
    """ОФИЦИАЛЬНО (ответ эксперта): сравниваем АМПЛИТУДЫ |K|, каждое изображение нормируется на max=1"""
    A = torch.sqrt(I + 1e-12)
    return A / A.amax((-2, -1), keepdim=True)

def evaluate(phi, T, name, dt):
    k, phq = quantize_phase(phi.detach())
    I = fresnel(phq)[0]
    An = to_image(I)
    An8 = torch.round(An * 255) / 255          # как будет в 8-битном png
    p = pcc(An8, T); s = ssim(An8, T)
    print(f"{name:15s} | PCC " + " ".join(f"{v:.3f}" for v in p.tolist())
          + f" | ср {p.mean():.3f} | SSIM " + " ".join(f"{v:.3f}" for v in s.tolist())
          + f" | ср {s.mean():.3f} | {dt*1e3:8.1f} мс")
    return k, An8

# ---------------- методы ----------------
def superposition(T):
    A = T.to(torch.complex64)               # T уже амплитуда
    rnd = torch.exp(2j * torch.pi * torch.rand(T.shape, device=DEV))
    return torch.angle(back(A * rnd).sum(0))

def gs(T, iters=300):
    A = T
    phi = superposition(T)
    for _ in range(iters):
        f = fresnel_field(phi)
        f = A * torch.exp(1j * torch.angle(f))   # амплитуду -> целевая, фазу оставляем
        phi = torch.angle(back(f).sum(0))
    return phi

def loss_fn(I, T):
    An = to_image(I)
    return (1 - pcc(An, T)).mean() + (1 - ssim(An, T)).mean()

def adam(T, iters=1500, lr=0.05, init=None):
    phi = (superposition(T) if init is None else init.clone()).requires_grad_(True)
    opt = torch.optim.Adam([phi], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        loss = loss_fn(fresnel(phi)[0], T)
        loss.backward()
        opt.step()
    return phi.detach()

# ---------------- тестовые картинки ----------------
def make_test_images():
    imgs = []
    # 1) натуральное фото из matplotlib
    import matplotlib.cbook as cb
    with cb.get_sample_data("grace_hopper.jpg") as f:
        im = Image.open(f).convert("L").resize((N, N), Image.BICUBIC)
    imgs.append(np.asarray(im))
    # 2) три окружности (как на слайде)
    im = Image.new("L", (N, N), 0); d = ImageDraw.Draw(im)
    for cx in (40, 64, 88):
        d.ellipse([cx - 22, 42, cx + 22, 86], outline=255, width=4)
    imgs.append(np.asarray(im))
    # 3) текст
    im = Image.new("L", (N, N), 0); d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
    except Exception:
        font = ImageFont.load_default()
    d.text((10, 44), "MEPhI", fill=255, font=font)
    imgs.append(np.asarray(im))
    # 4) градиент + фигуры
    yy, xx = np.mgrid[0:N, 0:N]
    g = (xx / N * 180).astype(np.uint8)
    im = Image.fromarray(g); d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 60, 60], fill=255); d.polygon([(90, 30), (115, 100), (65, 100)], fill=230)
    imgs.append(np.asarray(im))
    return imgs

def save_png(a, path):
    Image.fromarray(np.asarray(a, dtype=np.uint8)).save(path)

if __name__ == "__main__":
    report_geometry()
    if len(sys.argv) == 5:
        imgs = [np.asarray(Image.open(p).convert("L").resize((N, N))) for p in sys.argv[1:]]
    else:
        imgs = make_test_images()
    T = torch.tensor(np.stack(imgs), dtype=torch.float32, device=DEV) / 255.0  # (4,N,N) целевая АМПЛИТУДА
    T = T / T.amax((-2, -1), keepdim=True)                                       # нормировка на max=1

    os.makedirs("out", exist_ok=True)
    for i, im in enumerate(imgs):
        save_png(im, f"out/target_{i}.png")

    print()
    results = {}
    t = time.perf_counter(); phi = superposition(T); dt = time.perf_counter() - t
    results["superposition"] = evaluate(phi, T, "superposition", dt)
    t = time.perf_counter(); phi = gs(T, 300); dt = time.perf_counter() - t
    results["gs300"] = evaluate(phi, T, "GS x300", dt)
    phi_gs = phi
    t = time.perf_counter(); phi = adam(T, 1500); dt = time.perf_counter() - t
    results["adam1500"] = evaluate(phi, T, "Adam x1500", dt)
    t = time.perf_counter(); phi2 = adam(T, 1500, init=phi_gs); dt = time.perf_counter() - t
    results["gs+adam"] = evaluate(phi2, T, "GS->Adam x1500", dt)

    # для сравнения: насколько дорого стоят 4 плоскости — оптимизируем только под 1 плоскость
    w = torch.zeros(4); w[0] = 1
    phi1 = superposition(T).requires_grad_(True); opt = torch.optim.Adam([phi1], lr=0.05)
    for _ in range(1500):
        opt.zero_grad(); I = fresnel(phi1)[0][0:1]
        l = loss_fn(I, T[0:1]); l.backward(); opt.step()
    _, phq = quantize_phase(phi1.detach()); A = to_image(fresnel(phq)[0][0:1])
    print(f"\n[справка] только 1 плоскость (z=24.1): PCC {pcc(A, T[0:1]).item():.3f}, SSIM {ssim(A, T[0:1]).item():.3f}")

    k, rec = results["adam1500"]
    save_png(k.cpu().numpy(), "out/doe_adam.png")
    for i in range(4):
        save_png((rec[i] * 255).round().cpu().numpy(), f"out/recon_adam_{i}.png")

    # картинка-сводка
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [("цель", T)] + [(n, results[n][1]) for n in ["superposition", "gs300", "adam1500"]]
    fig, ax = plt.subplots(len(rows), 5, figsize=(12, 2.5 * len(rows)))
    for r, (n, imgsr) in enumerate(rows):
        for i in range(4):
            ax[r, i].imshow(imgsr[i].cpu(), cmap="gray", vmin=0, vmax=1); ax[r, i].axis("off")
            ax[r, i].set_title(f"{n} z={Z[i]*1e3:.1f}мм", fontsize=8)
        ax[r, 4].axis("off")
    ax[3, 4].imshow(k.cpu(), cmap="twilight"); ax[3, 4].set_title("ДОЭ (Adam)", fontsize=8)
    plt.tight_layout(); plt.savefig("out/summary.png", dpi=110)
    print("\nсохранено в out/: doe_adam.png, recon_adam_*.png, summary.png")
