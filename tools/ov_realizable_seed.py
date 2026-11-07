"""Почему GS на реализуемом наборе дал 1.0: проверяем, что совпала случайная фаза старта с той, которой набор генерировался."""
import sys
sys.path.insert(0, sys.argv[1])
import torch
from physics import Fresnel, evaluate, gs, gs_plus, quantize_phase, quantize_img, to_image, pick_device
dev = pick_device(); f = Fresnel().to(dev)
Ti = torch.load("data/val_div2k_indep_test777.pt").to(dev); Tr = torch.load("data/val_realizable_test777.pt").to(dev)
torch.manual_seed(0)
with torch.no_grad():
    phi = gs(Ti, f, iters=50); _, phq = quantize_phase(phi); Tr2 = quantize_img(to_image(f(phq)))
print("пересоздание набора с seed 0: max |разница| =", float((Tr2 - Tr).abs().max()))
for seed in [0, 1, 2, 3]:
    torch.manual_seed(seed); p, s, _ = evaluate(gs(Tr, f, iters=300), Tr, f)
    print(f"GS x300 на реализуемых, случайная фаза seed {seed}: КК {p.mean():.3f}  SSIM {s.mean():.3f}")
