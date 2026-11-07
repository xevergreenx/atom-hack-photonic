"""Бюджет разделения плоскостей: насколько картинка, сфокусированная в z1, остаётся видна в z2..z4,
если ДОЭ может быть комплексным (без ограничения phase-only). Зависит только от физики и фазы в плоскости изображения."""
import sys
sys.path.insert(0, sys.argv[1])
import torch
from physics import Fresnel, pcc, ssim, to_image, Z
torch.manual_seed(0)
f = Fresnel()
N = 128
# 1) ПСФ дефокуса: точка в центре плоскости 1, полная апертура
d = torch.zeros(1, 4, N, N, dtype=torch.complex64); d[0, 0, N // 2, N // 2] = 1
U = f.adjoint(d)[:, 0]                       # поле в плоскости ДОЭ (только от плоскости 1)
F = f.field(U)                               # (1,4,N,N)
I = (F.abs() ** 2)[0]
yy, xx = torch.meshgrid(torch.arange(N) - N // 2, torch.arange(N) - N // 2, indexing="ij")
r = torch.sqrt((yy ** 2 + xx ** 2).float())
print("ПСФ точки из z1 (полная апертура): радиус, внутри которого 50% / 80% энергии, пикс")
for j in range(4):
    e = I[j] / I[j].sum()
    rs = torch.sort(r.flatten()).values; order = torch.argsort(r.flatten())
    c = torch.cumsum(e.flatten()[order], 0)
    r50 = rs[(c >= 0.5).nonzero()[0]].item(); r80 = rs[(c >= 0.8).nonzero()[0]].item()
    print(f"  z{j+1} = {Z[j]*1e3:.1f} мм: r50 = {r50:.1f}, r80 = {r80:.1f}")
# 2) натуральные картинки: фаза в плоскости изображения нулевая / случайная
data = torch.load(sys.argv[2])
T = data if torch.is_tensor(data) else (data.get("T") if isinstance(data, dict) else data[0])
T = T.float(); T = T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)
T1 = T[:, 0]                                  # картинка плоскости 1, (B,N,N)
B = T1.shape[0]
for name, psi in [("нулевая (гладкая)", torch.zeros_like(T1)), ("случайная (диффузная)", 2 * torch.pi * torch.rand_like(T1))]:
    g = torch.zeros(B, 4, N, N, dtype=torch.complex64); g[:, 0] = T1 * torch.exp(1j * psi)
    U = f.adjoint(g)[:, 0]                   # комплексный ДОЭ, без ограничения phase-only
    A = to_image(f.field(U))                 # (B,4,N,N)
    P = pcc(A, T1[:, None].expand_as(A)).mean(0); Sm = ssim(A, T1[:, None].expand_as(A)).mean(0)
    print(f"фаза {name}: КК / SSIM картинки z1 с её же следом в z1..z4: " + "  ".join(f"{P[j]:.2f}/{Sm[j]:.2f}" for j in range(4)))
