"""Предыскажение целей в базисе Фурье: T' = |IFFT(FFT(T) * G(|f|))|, G — усиления по 6 радиальным полосам частот.
G ищется перебором (покоординатный поиск) на 16 четвёрках валидации (seed 123) по КК + SSIM после GS+ x100.
Потом фиксируется и проверяется на свежем наборе (seed 777): GS+ x300 по T' против GS+ x300 по T. Оценка — всегда против исходных T."""
import sys, time
sys.path.insert(0, sys.argv[1])
import torch
from physics import Fresnel, evaluate, gs_plus, gs, pick_device
dev = pick_device(); f = Fresnel().to(dev)

def load(p):
    d = torch.load(p); T = (d if torch.is_tensor(d) else d["T"]).float()
    return (T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)).to(dev)

N = 128
fy = torch.fft.fftfreq(N, device=dev); r = torch.sqrt(fy[:, None] ** 2 + fy[None, :] ** 2)
EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.35, 1.0]
BINS = [((r >= a) & (r < b)).float() for a, b in zip(EDGES[:-1], EDGES[1:])]

def prefilter(T, g):
    G = sum(gi * m for gi, m in zip(g, BINS))
    Tp = torch.fft.ifft2(torch.fft.fft2(T) * G).real.clamp_min(0)
    return Tp / Tp.amax((-2, -1), keepdim=True).clamp_min(1e-8)

def score(T, g, iters, kern=gs_plus, seed=0):
    torch.manual_seed(seed)
    phi = kern(prefilter(T, g), f, iters=iters)
    p, s, _ = evaluate(phi, T, f)
    return p.mean().item(), s.mean().item(), p.min(1).values.mean().item()

search = load(sys.argv[2])[:16]
g = [1.0] * 6
best = sum(score(search, g, 100)[:2]); print(f"старт g=1: КК+SSIM = {best:.4f}")
t0 = time.time()
for ps in range(2):
    for b in range(1, 6):                       # полоса 0 (DC) не трогаем: масштаб убирает нормировка
        for v in [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]:
            gt = list(g); gt[b] = v
            sc = sum(score(search, gt, 100)[:2])
            if sc > best + 1e-4: best, g = sc, gt
    print(f"проход {ps+1}: g = {[round(x, 2) for x in g]}, КК+SSIM = {best:.4f}  ({time.time()-t0:.0f} c)")

for path in sys.argv[3:]:
    T = load(path); print(path.split("/")[-1], T.shape[0], "четвёрок — КК / SSIM / КК худшей плоскости")
    for kn, kern in [("GS+ x300", gs_plus), ("GS x300", gs)]:
        a = score(T, [1.0] * 6, 300, kern); b_ = score(T, g, 300, kern)
        print(f"  {kn:9s} исходные цели: {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f}   предыскажённые: {b_[0]:.3f} / {b_[1]:.3f} / {b_[2]:.3f}")
