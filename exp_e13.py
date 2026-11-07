"""
Э13 (Г17): влияет ли начальная фаза в плоскостях изображений на качество GS и гибрида.

Старт = суперпозиция: каждая картинка с фазой ψ возвращается в плоскость ДОЭ, поля складываются, берётся фаза.
Варианты ψ:
  random        — равномерный шум (как во всех прошлых экспериментах)
  zero          — нулевая фаза
  quad_c        — «линзовая» квадратичная фаза ψ = c·π·r², r — радиус, нормированный на полкадра (c = 2, 8, 32)
  smooth_s      — сглаженный шум: шум, размытый гауссом с σ = s пикселей, растянутый на ±π·4
Метрики — на трёх наборах Э12 (срезы, независимые фото, реализуемые). Методы: GS x50, GS x300, гибрид GS x20 -> Adam x10.

Запуск: python3 -I exp_e13.py
"""
import os, sys, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
import torch.nn.functional as F
from physics import Fresnel, evaluate, ssim_sk, loss_fn, gs, N, pick_device

torch.manual_seed(0)
DEV = pick_device()
f = Fresnel().to(DEV)
sets = {k: torch.load(os.path.join(ROOT, "data", p)).to(DEV)
        for k, p in [("slices", "val_div2k.pt"), ("indep", "val_div2k_indep.pt"), ("realizable", "val_realizable.pt")]}

yy, xx = torch.meshgrid(torch.arange(N, device=DEV) - N / 2, torch.arange(N, device=DEV) - N / 2, indexing="ij")
r2 = (xx ** 2 + yy ** 2) / (N / 2) ** 2


def gauss_blur(x, s):
    k = int(4 * s) | 1
    g = torch.exp(-((torch.arange(k, device=DEV) - k // 2) ** 2) / (2 * s * s)); g = g / g.sum()
    x = x.reshape(-1, 1, N, N)
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="circular"), g.view(1, 1, 1, k))
    x = F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="circular"), g.view(1, 1, k, 1))
    return x


def init_phase(kind, shape):
    B = shape[0]
    if kind == "random":
        return 2 * torch.pi * torch.rand(shape, device=DEV)
    if kind == "zero":
        return torch.zeros(shape, device=DEV)
    if kind.startswith("quad_"):
        c = float(kind.split("_")[1])
        return (c * torch.pi * r2).expand(shape)
    if kind.startswith("smooth_"):
        s = float(kind.split("_")[1])
        n = gauss_blur(torch.randn(shape, device=DEV), s).reshape(shape)
        return n / n.std((-2, -1), keepdim=True) * torch.pi
    raise ValueError(kind)


def start(T, kind):
    psi = init_phase(kind, T.shape)
    H = f.adjoint(T * torch.exp(1j * psi)).sum(1)
    return torch.atan2(H.imag, H.real)


def hybrid(T, phi0, k=10, lr=0.1):
    with torch.no_grad():
        phi = gs(T, f, iters=20, phi0=phi0)
    p = phi.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(k):
        opt.zero_grad(); l, _, _ = loss_fn(p, T, f); l.backward(); opt.step()
    return p.detach()


def score(phi, T):
    p, s, rec = evaluate(phi, T, f)
    return dict(pcc=round(p.mean().item(), 4), ssim=round(s.mean().item(), 4), ssim_sk=round(ssim_sk(rec, T).mean().item(), 4))


kinds = ["random", "zero", "quad_2", "quad_8", "quad_32", "smooth_2", "smooth_8"]
res = {}
for kind in kinds:
    res[kind] = {}
    for sname, T in sets.items():
        torch.manual_seed(1)
        phi0 = start(T, kind)
        with torch.no_grad():
            g50 = gs(T, f, iters=50, phi0=phi0)
            g300 = gs(T, f, iters=250, phi0=g50)
        res[kind][sname] = {"start": score(phi0, T), "GS x50": score(g50, T), "GS x300": score(g300, T),
                            "гибрид": score(hybrid(T, phi0), T)}
    print("готово:", kind, flush=True)

with open(os.path.join(ROOT, "out", "e13_results.json"), "w") as fo:
    json.dump(res, fo, indent=1, ensure_ascii=False)

for sname in sets:
    print(f"\n== {sname} ==   (КК / SSIM)")
    print(f"{'старт':10s} | {'сам старт':>13s} | {'GS x50':>13s} | {'GS x300':>13s} | {'гибрид':>13s}")
    for kind in kinds:
        r = res[kind][sname]
        cells = " | ".join(f"{r[m]['pcc']:.3f} / {r[m]['ssim']:.3f}" for m in ["start", "GS x50", "GS x300", "гибрид"])
        print(f"{kind:10s} | {cells}")
