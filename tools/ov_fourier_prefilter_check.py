"""Фиксированное фурье-предыскажение целей (найдено перебором на валидации seed 123) на свежих наборах, seed старта 1."""
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
from PIL import Image
from physics import Fresnel, evaluate, gs, gs_plus, loss_fn, pick_device
dev = pick_device(); f = Fresnel().to(dev)
N = 128
fy = torch.fft.fftfreq(N, device=dev); r = torch.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
E = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 1.0]; G0 = [1.0, 1.25, 1.5, 1.5, 2.0, 1.5]
G = sum(g * ((r >= a) & (r < b)).float() for g, a, b in zip(G0, E[:-1], E[1:]))
def pre(T):
    Tp = torch.fft.ifft2(torch.fft.fft2(T) * G).real.clamp_min(0); return Tp / Tp.amax((-2, -1), keepdim=True).clamp_min(1e-8)
def norm(T): return T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)
def hybrid(Tg, T):                      # GS+ x20 по Tg -> Adam x10 по исходным T (метрики жюри)
    with torch.no_grad(): phi = gs_plus(Tg, f, iters=20)
    p = phi.clone().requires_grad_(True); o = torch.optim.Adam([p], lr=0.1)
    for _ in range(10):
        o.zero_grad(); l, _, _ = loss_fn(p, T, f); l.backward(); o.step()
    return p.detach()
D = sys.argv[2]; cs = sorted(os.listdir(os.path.join(D, "previews")))
Td = norm(torch.stack([torch.from_numpy(np.stack([np.asarray(Image.open(os.path.join(D, "previews", c, f"target_{j}.png")).convert("L")) for j in range(1, 5)]).astype(np.float32) / 255) for c in cs]).to(dev))
sets = [("независимые фото (777)", norm(torch.load("data/val_div2k_indep_test777.pt").to(dev))),
        ("срезы (777)", norm(torch.load("data/val_div2k_test777.pt").to(dev))),
        ("реализуемые (777)", norm(torch.load("data/val_realizable_test777.pt").to(dev))),
        ("6 превью DUSFN", Td)]
meth = [("GS x300", lambda Tg, T: gs(Tg, f, iters=300)), ("GS+ x300", lambda Tg, T: gs_plus(Tg, f, iters=300)), ("GS+ x20 -> Adam x10", hybrid)]
print("КК / SSIM / КК худшей плоскости; слева — исходные цели, справа — предыскажённые")
for sn, T in sets:
    print(sn)
    for mn, m in meth:
        out = []
        for Tg in (T, pre(T)):
            torch.manual_seed(1); p, s, _ = evaluate(m(Tg, T), T, f)
            out.append(f"{p.mean():.3f} / {s.mean():.3f} / {p.min(1).values.mean():.3f}")
        print(f"  {mn:20s} {out[0]}   ->   {out[1]}")
