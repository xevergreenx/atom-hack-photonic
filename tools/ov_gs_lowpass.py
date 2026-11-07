"""GS, где фаза в плоскостях изображения на каждой итерации сглаживается окном Гаусса sigma (пикс).
sigma = 0 — обычный GS. Ручка «диффузность»: от спекла с разделением плоскостей до гладкого без разделения."""
import sys, time
sys.path.insert(0, sys.argv[1])
import torch, torch.nn.functional as Fn
from physics import Fresnel, evaluate, superposition, pick_device
dev = pick_device(); f = Fresnel().to(dev); torch.manual_seed(0)

def blur_c(x, s):
    if s <= 0: return x
    k = int(3 * s) * 2 + 1; t = torch.arange(k, device=x.device, dtype=torch.float32) - k // 2
    g = torch.exp(-t ** 2 / (2 * s * s)); g = (g / g.sum())
    w = (g[:, None] * g[None])[None, None]
    sh = x.shape; xr = torch.view_as_real(x).permute(0, 1, 4, 2, 3).reshape(-1, 1, sh[-2], sh[-1])
    y = Fn.conv2d(Fn.pad(xr, (k // 2,) * 4, mode="circular"), w)
    y = y.reshape(sh[0], sh[1], 2, sh[-2], sh[-1]).permute(0, 1, 3, 4, 2).contiguous()
    return torch.view_as_complex(y)

def gs_lp(T, s, iters=300):
    phi = superposition(T, f)
    for _ in range(iters):
        F = f(phi); u = F / (F.abs() + 1e-12)
        u = blur_c(u, s); u = u / (u.abs() + 1e-12)
        phi = torch.angle(f.adjoint(T * u).sum(1))
    return phi

for path in sys.argv[2:]:
    d = torch.load(path); T = (d if torch.is_tensor(d) else d["T"]).float().to(dev)
    T = T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)
    print(path.split("/")[-1])
    for s in [0, 0.7, 1, 1.5, 2, 3, 5]:
        t0 = time.time(); phi = gs_lp(T, s)
        p, q, _ = evaluate(phi, T, f)
        print(f"  sigma={s:<4} КК {p.mean():.3f}  КК-худш {p.min(1).values.mean():.3f}  SSIM {q.mean():.3f}  SSIM-худш {q.min(1).values.mean():.3f}  ({time.time()-t0:.1f} c)")
