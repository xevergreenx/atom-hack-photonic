"""Без нейросети: старт = суперпозиция с НУЛЕВОЙ фазой в плоскостях (как у необученной DUSFN) -> K шагов Adam по метрикам.
Сравнение со стартом от случайной фазы и от GS+ x20. Наборы: срезы, независимые фото, 6 превью DUSFN."""
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
from PIL import Image
from physics import Fresnel, evaluate, loss_fn, superposition, gs_plus, pick_device
dev = pick_device(); f = Fresnel().to(dev)

def load_pt(p):
    d = torch.load(p); T = (d if torch.is_tensor(d) else d["T"]).float()
    return (T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)).to(dev)

def load_dusfn(D):
    out, does = [], []
    for c in sorted(os.listdir(os.path.join(D, "previews"))):
        T = np.stack([np.asarray(Image.open(os.path.join(D, "previews", c, f"target_{j}.png")).convert("L")) for j in range(1, 5)])
        out.append(torch.from_numpy(T.astype(np.float32) / 255))
        k = np.asarray(Image.open(os.path.join(D, "previews", c, "doe_phase_uint8.png"))).astype(np.float32)
        does.append(torch.from_numpy(k / 255 * 2 * np.pi - np.pi))
    T = torch.stack(out); T = T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)
    return T.to(dev), torch.stack(does).to(dev)

def adam(phi0, T, K, wp, ws, lr=0.05):
    p = phi0.clone().requires_grad_(True); o = torch.optim.Adam([p], lr=lr)
    for _ in range(K):
        o.zero_grad(); l, _, _ = loss_fn(p, T, f, w_pcc=wp, w_ssim=ws); l.backward(); o.step()
    return p.detach()

def rep(name, phi, T):
    p, s, _ = evaluate(phi, T, f)
    print(f"  {name:38s} КК {p.mean():.3f}  КК-худш {p.min(1).values.mean():.3f}  SSIM {s.mean():.3f}  SSIM-худш {s.min(1).values.mean():.3f}")

sets = [("срезы", load_pt(sys.argv[2]), None), ("независимые фото", load_pt(sys.argv[3]), None)]
Td, Dd = load_dusfn(sys.argv[4]); sets.append(("6 превью DUSFN", Td, Dd))
torch.manual_seed(0)
for name, T, D in sets:
    print(name, T.shape[0])
    starts = {"нулевая фаза (гладкий)": torch.angle(f.adjoint(T.to(torch.complex64)).sum(1)),
              "случайная фаза": superposition(T, f),
              "GS+ x20": gs_plus(T, f, iters=20)}
    if D is not None:
        rep("DUSFN как есть", D, T); starts["маска DUSFN"] = D
    for sn, phi0 in starts.items():
        rep(f"{sn}, K=0", phi0, T)
        for K in (15, 50, 200):
            rep(f"{sn} -> Adam x{K} (1:1)", adam(phi0, T, K, 1.0, 1.0), T)
        rep(f"{sn} -> Adam x50 (КК:SSIM=1:2)", adam(phi0, T, 50, 0.5, 1.0), T)
