"""
Э8: можно ли быстро поднять SSIM выше плато GS (~0.17)?
Старт (суперпозиция / сеть A из M-2 / GS x10 / GS x20 / сеть A + GS x10) -> K шагов Adam прямо по метрикам (L-0).
Без обучения сетей. Валидация — data/val_div2k.pt. Время — одна маска (батч 1) на MPS.

Запуск: python3 -I exp_e8.py
"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from physics import Fresnel, evaluate, loss_fn, superposition, gs, pick_device
from models import M2

torch.manual_seed(0)
DEV = pick_device()
f = Fresnel().to(DEV)
Tv = torch.load(os.path.join(ROOT, "data", "val_div2k.pt")).to(DEV)
m2 = M2().to(DEV); m2.load_state_dict(torch.load(os.path.join(ROOT, "runs", "s1_m2", "model.pt"))); m2.eval()


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def init(name, T):
    with torch.no_grad():
        if name == "superposition":
            return superposition(T, f)
        if name == "netA":
            return m2(T, f)
        if name == "gs10":
            return gs(T, f, iters=10)
        if name == "gs20":
            return gs(T, f, iters=20)
        if name == "netA+gs10":
            return gs(T, f, iters=10, phi0=m2(T, f))


def refine(phi0, T, k, lr):
    p = phi0.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(k):
        opt.zero_grad(); l, _, _ = loss_fn(p, T, f); l.backward(); opt.step()
    return p.detach()


def score(phi):
    p, s, _ = evaluate(phi, Tv, f)
    return round(p.mean().item(), 4), round(s.mean().item(), 4), round(p.amin(1).mean().item(), 4), round(s.amin(1).mean().item(), 4)


def latency(name, k, lr, reps=5):
    T1 = Tv[:1]
    init(name, T1); refine(init(name, T1), T1, min(k, 2), lr); sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); phi = init(name, T1)
        if k:
            phi = refine(phi, T1, k, lr)
        sync(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2]


res = []
for name in ["superposition", "netA", "gs10", "gs20", "netA+gs10"]:
    phi0 = init(name, Tv)
    pc, ss, pm, sm = score(phi0)
    res.append(dict(init=name, k=0, lr=0, pcc=pc, ssim=ss, pcc_min=pm, ssim_min=sm))
    for lr in [0.05, 0.1, 0.2]:
        for k in [5, 10, 20, 50]:
            pc, ss, pm, sm = score(refine(phi0, Tv, k, lr))
            res.append(dict(init=name, k=k, lr=lr, pcc=pc, ssim=ss, pcc_min=pm, ssim_min=sm))
    print(f"готово: {name}", flush=True)

# время для лучших lr по каждому k
best = {}
for r in res:
    key = (r["init"], r["k"])
    if key not in best or r["ssim"] > best[key]["ssim"]:
        best[key] = r
for key, r in best.items():
    r["latency_ms"] = round(latency(r["init"], r["k"], r["lr"] or 0.1) * 1e3, 1)

with open(os.path.join(ROOT, "out", "e8_results.json"), "w") as fo:
    json.dump(dict(all=res, best=[best[k] for k in best]), fo, indent=1, ensure_ascii=False)

print(f"\nЭ8 (лучший lr для каждого старта и K), {len(Tv)} четвёрок DIV2K, устройство {DEV}")
print(f"{'старт':16s} {'K':>3s} {'lr':>5s} | {'КК':>6s} {'КК мин':>6s} | {'SSIM':>6s} {'SSIM мин':>8s} | время MPS")
for (name, k), r in best.items():
    print(f"{name:16s} {k:3d} {r['lr']:5.2f} | {r['pcc']:6.3f} {r['pcc_min']:6.3f} | {r['ssim']:6.3f} {r['ssim_min']:8.3f} | {r['latency_ms']:7.1f} мс")
