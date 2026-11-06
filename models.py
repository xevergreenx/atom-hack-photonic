"""
Нейросетевые энкодеры: 4 изображения (B,4,N,N) -> фаза ДОЭ (B,N,N).
Декодер у всех один — фиксированный оператор Френеля (physics.Fresnel), он не обучается.

M-1  «в лоб»:            U-Net: изображения -> (cos, sin) -> atan2.
M-2  «с физикой внутри»:  сеть A предсказывает фазу ψ_k для каждой плоскости ->
                          точный сопряжённый Френель каждой плоскости в плоскость ДОЭ -> взвешенная сумма ->
                          сеть B (в плоскости ДОЭ) дочищает -> фаза.  (≈ DeepCGH / идея 1 коллеги)
M-5  M-2 + K обучаемых шагов уточнения: Френель -> GS-невязка по плоскостям -> сопряжённый Френель ->
                          маленькая сеть -> обновление фазы. Веса фиксированы, всё внутри одного прохода.
"""
import torch
import torch.nn as nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
    )


class UNet(nn.Module):
    """U-Net со «стеблем» PixelUnshuffle(2): 128x128 -> 64x64 с 4x каналов без потери информации,
    в конце PixelShuffle обратно. На MPS это в ~4 раза быстрее при том же числе параметров."""

    def __init__(self, cin, cout, base=32, depth=4, zero_out=False, ps=2):
        super().__init__()
        self.ps = ps
        cin, cout_in = cin * ps * ps, cout * ps * ps
        ch = [base * 2 ** i for i in range(depth)]
        self.inc = _block(cin, ch[0])
        self.down = nn.ModuleList([_block(ch[i], ch[i + 1]) for i in range(depth - 1)])
        self.up = nn.ModuleList([_block(ch[i + 1] + ch[i], ch[i]) for i in reversed(range(depth - 1))])
        self.pool = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out = nn.Conv2d(ch[0], cout_in, 1)
        if zero_out:
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        if self.ps > 1:
            x = nn.functional.pixel_unshuffle(x, self.ps)
        skips = [self.inc(x)]
        for d in self.down:
            skips.append(d(self.pool(skips[-1])))
        h = skips.pop()
        for u in self.up:
            h = u(torch.cat([self.upsample(h), skips.pop()], 1))
        h = self.out(h)
        return nn.functional.pixel_shuffle(h, self.ps) if self.ps > 1 else h


def _rms(z, dims=(-2, -1)):
    return torch.sqrt((z.real ** 2 + z.imag ** 2).mean(dims, keepdim=True) + 1e-12)


class M1(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.net = UNet(4, 2, base)

    def forward(self, T, fres):
        o = self.net(T)
        return torch.atan2(o[:, 1], o[:, 0])


class M2(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.netA = UNet(4, 8, base)                            # (cos, sin) для 4 плоскостей
        self.netB = UNet(10, 2, base, zero_out=True)            # поправка в плоскости ДОЭ, на старте 0
        self.w = nn.Parameter(torch.ones(4, 2) * torch.tensor([1.0, 0.0]))   # комплексные веса плоскостей

    def encode(self, T, fres):
        o = self.netA(T).view(T.shape[0], 4, 2, *T.shape[-2:])
        psi = torch.atan2(o[:, :, 1], o[:, :, 0])               # (B,4,N,N)
        Bk = fres.adjoint(T * torch.exp(1j * psi))              # (B,4,N,N) в плоскости ДОЭ
        w = torch.complex(self.w[:, 0], self.w[:, 1])[None, :, None, None]
        H = (w * Bk).sum(1)
        Hn = H / _rms(H)
        Bn = Bk / _rms(Bk, dims=(-3, -2, -1))
        x = torch.cat([Hn.real[:, None], Hn.imag[:, None], Bn.real, Bn.imag], 1)
        c = self.netB(x)
        return torch.atan2(Hn.imag + c[:, 1], Hn.real + c[:, 0])

    def forward(self, T, fres):
        return self.encode(T, fres)


class M5(M2):
    def __init__(self, base=24, base_r=16, steps=3):
        super().__init__(base)
        self.netR = UNet(12, 2, base_r, zero_out=True)          # общая для всех шагов
        self.a = nn.Parameter(torch.full((steps,), 0.5))        # вес GS-поправки на каждом шаге
        self.steps = steps

    def forward(self, T, fres):
        phi = self.encode(T, fres)
        for t in range(self.steps):
            F = fres(phi)                                       # (B,4,N,N)
            A = torch.sqrt(F.real ** 2 + F.imag ** 2 + 1e-12)
            s = A.amax((-2, -1), keepdim=True)                  # масштаб под нормировку на max
            Ft = T * s * (F / A)                                # GS-проекция: амплитуда цели, фаза текущая
            D = fres.adjoint(Ft - F)                            # невязки, перенесённые в плоскость ДОЭ
            S = D.sum(1)
            Sn = S / _rms(S)
            Dn = D / _rms(D, dims=(-3, -2, -1))
            x = torch.cat([torch.cos(phi)[:, None], torch.sin(phi)[:, None],
                           Sn.real[:, None], Sn.imag[:, None], Dn.real, Dn.imag], 1)
            c = self.netR(x)
            zr = torch.cos(phi) + self.a[t] * Sn.real + c[:, 0]
            zi = torch.sin(phi) + self.a[t] * Sn.imag + c[:, 1]
            phi = torch.atan2(zi, zr)
        return phi


MODELS = {"m1": M1, "m2": M2, "m5": M5}


def n_params(m):
    return sum(p.numel() for p in m.parameters())
