"""
Общий модуль: физика (Френель с одним БПФ), метрики и лосс — одинаковые для всех экспериментов (Р9).
Всё батчевое: фаза (B,N,N), поля и изображения (B,4,N,N).

Официальная схема оценки (ответ экспертов): сравниваем АМПЛИТУДЫ |K|, каждое изображение
нормируется на max = 1. Фаза ДОЭ и изображения квантуются в 8 бит.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N = 128
DX = 10e-6
LAM = 532e-9
Z = (24.1e-3, 25.3e-3, 26.5e-3, 27.7e-3)


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Fresnel(nn.Module):
    """Прямой и сопряжённый операторы для 4 плоскостей. Оператор унитарный (Э3): adjoint == inverse."""

    def __init__(self):
        super().__init__()
        c = (torch.arange(N, dtype=torch.float64) - N / 2) * DX
        uu, vv = torch.meshgrid(c, c, indexing="ij")
        r2 = uu ** 2 + vv ** 2
        ch = torch.stack([torch.exp(1j * torch.pi / (LAM * z) * r2) for z in Z]).to(torch.complex64)
        self.register_buffer("chirp", ch)                     # (4,N,N)

    @staticmethod
    def _fft(x):
        return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1)) / N

    @staticmethod
    def _ifft(x):
        return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1)) * N

    def forward(self, phi):
        """phi (B,N,N) -> комплексные поля в 4 плоскостях (B,4,N,N)"""
        return self._fft(torch.exp(1j * phi)[:, None] * self.chirp)

    def field(self, h):
        """комплексное поле в плоскости ДОЭ (B,N,N) или (B,4,N,N) -> (B,4,N,N)"""
        if h.dim() == 3:
            h = h[:, None]
        return self._fft(h * self.chirp)

    def adjoint(self, f):
        """поля в 4 плоскостях (B,4,N,N) -> 4 поля в плоскости ДОЭ (B,4,N,N)"""
        return self._ifft(f) * self.chirp.conj()


# ------------------------------------------------------------------ изображения и метрики
def to_image(field_or_I, is_field=True):
    """амплитуда, нормированная на max = 1 (схема жюри)"""
    if is_field:   # гладкий модуль: без бесконечного градиента в нуле
        A = torch.sqrt(field_or_I.real ** 2 + field_or_I.imag ** 2 + 1e-12)
    else:
        A = torch.sqrt(field_or_I.clamp_min(0) + 1e-12)
    return A / A.amax((-2, -1), keepdim=True).clamp_min(1e-12)


def pcc(a, b):
    a = a - a.mean((-2, -1), keepdim=True)
    b = b - b.mean((-2, -1), keepdim=True)
    return (a * b).sum((-2, -1)) / torch.sqrt((a * a).sum((-2, -1)) * (b * b).sum((-2, -1)) + 1e-12)


_GW = {}


def _gauss(device, win=11, sigma=1.5):
    key = str(device)
    if key not in _GW:
        g = torch.exp(-((torch.arange(win, dtype=torch.float32) - win // 2) ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        _GW[key] = (g[:, None] * g[None]).to(device)[None, None]
    return _GW[key]


def ssim(a, b):
    """классический SSIM (Wang 2004): окно Гаусса 11x11, sigma 1.5, c1=0.01^2, c2=0.03^2, данные в [0,1]"""
    shp = a.shape
    a = a.reshape(-1, 1, shp[-2], shp[-1])
    b = b.reshape(-1, 1, shp[-2], shp[-1])
    w = _gauss(a.device)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)
    saa = F.conv2d(a * a, w) - mu_a ** 2
    sbb = F.conv2d(b * b, w) - mu_b ** 2
    sab = F.conv2d(a * b, w) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.mean((-2, -1)).reshape(shp[:-2])


def ssim_sk(a, b, win=7):
    """SSIM как skimage.metrics.structural_similarity по умолчанию (data_range=1):
    равномерное окно 7x7, выборочная ковариация (N/(N-1)), края шириной 3 пикселя отбрасываются.
    Реализация жюри неизвестна — отчитываемся в обоих вариантах (Р13)."""
    shp = a.shape
    a = a.reshape(-1, 1, shp[-2], shp[-1])
    b = b.reshape(-1, 1, shp[-2], shp[-1])
    w = torch.ones(1, 1, win, win, device=a.device, dtype=a.dtype) / win ** 2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    cov = win * win / (win * win - 1)
    mu_a, mu_b = F.conv2d(a, w), F.conv2d(b, w)                 # «valid» = ровно область после обрезки краёв
    saa = cov * (F.conv2d(a * a, w) - mu_a ** 2)
    sbb = cov * (F.conv2d(b * b, w) - mu_b ** 2)
    sab = cov * (F.conv2d(a * b, w) - mu_a * mu_b)
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.mean((-2, -1)).reshape(shp[:-2])


def quantize_phase(phi):
    """8 бит как в ТЗ: 0 -> -pi, 255 -> +pi. Возвращает (коды uint8-значения, квантованную фазу)"""
    w = torch.remainder(phi + torch.pi, 2 * torch.pi)
    k = torch.round(w / (2 * torch.pi) * 255).clamp(0, 255)
    return k, k / 255 * 2 * torch.pi - torch.pi


def quantize_img(x):
    return torch.round(x.clamp(0, 1) * 255) / 255


@torch.no_grad()
def evaluate(phi, T, fres):
    """оценка как у жюри: фаза -> 8 бит -> восстановление -> амплитуда/max -> 8 бит -> КК и SSIM.
    Возвращает pcc (B,4), ssim (B,4) и восстановления (B,4,N,N)."""
    _, phq = quantize_phase(phi)
    rec = quantize_img(to_image(fres(phq)))
    return pcc(rec, T), ssim(rec, T), rec


# ------------------------------------------------------------------ лосс
def loss_fn(phi, T, fres, w_pcc=1.0, w_ssim=1.0, worst=0.0, charb=0.0):
    """
    L-0: mean[(1-КК) + (1-SSIM)] по плоскостям и батчу.
    worst>0  (L-1): + worst * mean по батчу от худшей плоскости
    charb>0  (L-3): + charb * Charbonnier(rec - T)
    w_ssim>1 (L-4): повышенный вес SSIM
    """
    rec = to_image(fres(phi))
    p, s = pcc(rec, T), ssim(rec, T)
    per = w_pcc * (1 - p) + w_ssim * (1 - s)                  # (B,4)
    loss = per.mean()
    if worst > 0:
        loss = loss + worst * per.amax(-1).mean()
    if charb > 0:
        loss = loss + charb * torch.sqrt((rec - T) ** 2 + 1e-6).mean()
    return loss, p.detach(), s.detach()


# ------------------------------------------------------------------ классические методы (батчевые)
def superposition(T, fres, gen=None):
    rnd = torch.rand(T.shape, device=T.device, generator=gen)
    f = T.to(torch.complex64) * torch.exp(2j * torch.pi * rnd)
    return torch.angle(fres.adjoint(f).sum(1))


def gs_plus(T, fres, iters=300, phi0=None, plane_w=True, feedback=True, fb_pow=0.5, fb_clip=(0.3, 3.0), callback=None):
    """GS с двумя добавками из Э14:
    plane_w  — вклад плоскости в сумму растёт, если её КК хуже среднего (подтягивает худшую плоскость);
    feedback — попиксельный вес целевой амплитуды растёт там, где восстановление недобирает яркость (SSIM)."""
    phi = superposition(T, fres) if phi0 is None else phi0
    W = torch.ones_like(T)
    for it in range(1, iters + 1):
        F = fres(phi)
        U = F / torch.sqrt(F.real ** 2 + F.imag ** 2 + 1e-12)
        A = to_image(F)
        if feedback:
            W = (W * ((T + 1e-3) / (A + 1e-3)) ** fb_pow).clamp(*fb_clip)
        Bk = fres.adjoint(T * W * U)
        if plane_w:
            w = (1.0 - pcc(A, T)).clamp_min(1e-3)
            Bk = Bk * (w / w.mean(1, keepdim=True))[..., None, None]
        G = Bk.sum(1)
        phi = torch.atan2(G.imag, G.real)
        if callback is not None:
            callback(it, phi)
    return phi


def gs(T, fres, iters=300, phi0=None, callback=None):
    """многоплоскостной (параллельный) Герчберг-Сэкстон. callback(it, phi) после каждой итерации."""
    phi = superposition(T, fres) if phi0 is None else phi0
    for it in range(1, iters + 1):
        f = fres(phi)
        f = T * torch.exp(1j * torch.angle(f))
        phi = torch.angle(fres.adjoint(f).sum(1))
        if callback is not None:
            callback(it, phi)
    return phi
