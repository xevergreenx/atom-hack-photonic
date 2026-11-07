"""
Э14 (Г18): снимается ли застревание GS модификацией итераций.

Варианты ядра (одна «итерация» = 4 прямых + 4 обратных распространения у всех вариантов):
  parallel        — обычный многоплоскостной GS (как раньше): все плоскости поправляются одновременно, поля суммируются
  sequential      — плоскости поправляются по очереди (ping-pong): маска обновляется после каждой плоскости
  plane_weighted  — parallel, но вклад плоскости во взвешенную сумму растёт, если её КК хуже среднего
  pixel_feedback  — parallel, но целевая амплитуда умножается на попиксельный вес W, который растёт там,
                    где восстановление недобирает яркость (W <- W * (T / Â)^0.5, ограничение [0.3, 3])
  overrelax_b     — parallel с «перелётом» в плоскости изображения: F' = P + b*(P - F), P = T*e^{i arg F} (b = 0.5, 1.0)

Старт у всех — одна и та же суперпозиция со случайной фазой. Наборы Э12. Метрики после 50 и 300 итераций.
Запуск: python3 -I exp_e14.py
"""
import os, sys, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from physics import Fresnel, evaluate, ssim_sk, superposition, pcc, to_image, pick_device

torch.manual_seed(0)
DEV = pick_device()
f = Fresnel().to(DEV)
sets = {k: torch.load(os.path.join(ROOT, "data", p)).to(DEV)
        for k, p in [("slices", "val_div2k.pt"), ("indep", "val_div2k_indep.pt"), ("realizable", "val_realizable.pt")]}


def unit(F):
    return F / torch.sqrt(F.real ** 2 + F.imag ** 2 + 1e-12)


def ang(H):
    return torch.atan2(H.imag, H.real)


def run(kind, T, phi, iters, cb):
    B = T.shape[0]
    W = torch.ones_like(T)
    for it in range(1, iters + 1):
        if kind == "sequential":
            for k in range(4):
                Fk = f._fft(torch.exp(1j * phi) * f.chirp[k])
                Gk = f._ifft(T[:, k] * unit(Fk)) * f.chirp[k].conj()
                phi = ang(Gk)
        else:
            F = f(phi)
            P = T * unit(F)
            if kind == "plane_weighted":
                q = pcc(to_image(F), T)                                  # (B,4)
                w = (1.0 - q).clamp_min(1e-3)
                w = (w / w.mean(1, keepdim=True))[..., None, None]
                G = (w * f.adjoint(P)).sum(1)
            elif kind == "pixel_feedback":
                A = to_image(F)
                W = (W * torch.sqrt((T + 1e-3) / (A + 1e-3))).clamp(0.3, 3.0)
                G = f.adjoint(T * W * unit(F)).sum(1)
            elif kind.startswith("overrelax_"):
                b = float(kind.split("_")[1])
                s = (F.abs().amax((-2, -1), keepdim=True) / T.amax((-2, -1), keepdim=True).clamp_min(1e-8))
                Ps = P * s                                                 # проекция в масштабе текущего поля
                G = f.adjoint(Ps + b * (Ps - F)).sum(1)
            else:
                G = f.adjoint(P).sum(1)
            phi = ang(G)
        cb(it, phi)
    return phi


def score(phi, T):
    p, s, rec = evaluate(phi, T, f)
    return dict(pcc=round(p.mean().item(), 4), pcc_min=round(p.amin(1).mean().item(), 4),
                ssim=round(s.mean().item(), 4), ssim_sk=round(ssim_sk(rec, T).mean().item(), 4))


kinds = ["parallel", "sequential", "plane_weighted", "pixel_feedback", "overrelax_0.5", "overrelax_1.0"]
res = {}
for kind in kinds:
    res[kind] = {}
    for sname, T in sets.items():
        torch.manual_seed(1)
        phi0 = superposition(T, f)
        out = {}
        def cb(it, phi):
            if it in (50, 300):
                out[it] = score(phi, T)
        with torch.no_grad():
            run(kind, T, phi0, 300, cb)
        res[kind][sname] = out
    print("готово:", kind, flush=True)

with open(os.path.join(ROOT, "out", "e14_results.json"), "w") as fo:
    json.dump(res, fo, indent=1, ensure_ascii=False)

for sname in sets:
    print(f"\n== {sname} ==   КК / КК мин / SSIM")
    print(f"{'ядро':15s} | {'x50':>21s} | {'x300':>21s}")
    for kind in kinds:
        r = res[kind][sname]
        c = " | ".join(f"{r[i]['pcc']:.3f} / {r[i]['pcc_min']:.3f} / {r[i]['ssim']:.3f}" for i in (50, 300))
        print(f"{kind:15s} | {c}")
