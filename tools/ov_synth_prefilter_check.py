"""Проверка встроенного --prefilter: тот же synthesize.build, что в скрипте сдачи, на свежих наборах и превью DUSFN."""
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
from PIL import Image
from physics import Fresnel, evaluate, pick_device
from synthesize import build
dev = pick_device(); f = Fresnel().to(dev)
def norm(T): return T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)
D = sys.argv[2]; cs = sorted(os.listdir(os.path.join(D, "previews")))
Td = norm(torch.stack([torch.from_numpy(np.stack([np.asarray(Image.open(os.path.join(D, "previews", c, f"target_{j}.png")).convert("L")) for j in range(1, 5)]).astype(np.float32) / 255) for c in cs]).to(dev))
sets = [("независимые фото 777", norm(torch.load("data/val_div2k_indep_test777.pt").to(dev))),
        ("срезы 777", norm(torch.load("data/val_div2k_test777.pt").to(dev))), ("превью DUSFN", Td)]
print("КК / SSIM / КК худшей плоскости")
for sn, T in sets:
    print(sn)
    for m in ("gs", "gsplus", "hybrid"):
        r = []
        for pf in (False, True):
            torch.manual_seed(1); p, s, _ = evaluate(build(m, 300, dev, pf)(T, f), T, f)
            r.append(f"{p.mean():.3f} / {s.mean():.3f} / {p.min(1).values.mean():.3f}")
        print(f"  {m:7s} {r[0]}  ->  {r[1]}")
