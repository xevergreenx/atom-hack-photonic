"""
Э17: итоговый график для защиты — качество (КК, SSIM) от времени синтеза одной маски, три набора.

Кривые: GS и GS+ (веса плоскостей + обратная связь) при K = 1…300 итераций.
Точки: M-2 (один проход), M-10, M-10p, гибриды «ядро x20 -> Adam xk».
Время — одна маска (батч 1) на MPS, медиана 5 замеров после прогрева. Качество — на 32 четвёрках каждого набора.

Запуск (на свободной видеокарте): python3 -I exp_e17.py
"""
import os, sys, json, time
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from physics import Fresnel, evaluate, loss_fn, gs, gs_plus, superposition, pick_device
from models import MODELS

torch.manual_seed(0)
DEV = pick_device()
f = Fresnel().to(DEV)
sets = {k: torch.load(os.path.join(ROOT, "data", p)).to(DEV)
        for k, p in [("slices", "val_div2k.pt"), ("indep", "val_div2k_indep.pt"), ("realizable", "val_realizable.pt")]}
KS = [1, 2, 5, 10, 20, 50, 100, 300]


def sync():
    if DEV == "mps":
        torch.mps.synchronize()


def timeit(fn, T1, reps=5):
    fn(T1); sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(T1); sync(); ts.append(time.perf_counter() - t)
    ts.sort()
    return ts[len(ts) // 2] * 1e3


def adam_after(core, g, k, lr=0.1):
    def run(T):
        with torch.no_grad():
            phi = core(T, f, iters=g)
        p = phi.clone().requires_grad_(True)
        opt = torch.optim.Adam([p], lr=lr)
        for _ in range(k):
            opt.zero_grad(); l, _, _ = loss_fn(p, T, f); l.backward(); opt.step()
        return p.detach()
    return run


def model(name, ckpt):
    path = os.path.join(ROOT, "runs", ckpt, "model.pt")
    if not os.path.exists(path):
        return None
    m = MODELS[name]().to(DEV); m.load_state_dict(torch.load(path, map_location=DEV)); m.eval()
    def run(T):
        with torch.no_grad():
            return m(T, f).detach()
    return run


def q(fn):
    out = {}
    for s, T in sets.items():
        p, ss, _ = evaluate(fn(T), T, f)
        out[s] = dict(pcc=round(p.mean().item(), 4), ssim=round(ss.mean().item(), 4), pcc_min=round(p.amin(1).mean().item(), 4))
    return out


T1 = sets["slices"][:1]
res = {"curves": {}, "points": {}}
for cname, core in [("GS", gs), ("GS+", gs_plus)]:
    pts = []
    for k in KS:
        fn = lambda T, k=k, core=core: core(T, f, iters=k)
        pts.append(dict(k=k, ms=round(timeit(fn, T1), 2), **q(fn)))
    res["curves"][cname] = pts
    print("кривая готова:", cname, flush=True)

points = {
    "M-2 (один проход)": model("m2", "e11_m2_long"),
    "M-10": model("m10", "s1_m10"),
    "M-10p": model("m10p", "f1_m10p_mix"),
    "GS+ x20 -> Adam x5": adam_after(gs_plus, 20, 5),
    "GS+ x20 -> Adam x10": adam_after(gs_plus, 20, 10),
    "GS+ x20 -> Adam x20": adam_after(gs_plus, 20, 20),
}
for name, fn in points.items():
    if fn is None:
        continue
    res["points"][name] = dict(ms=round(timeit(fn, T1), 2), **q(fn))
    print("точка готова:", name, flush=True)

with open(os.path.join(ROOT, "out", "e17_results.json"), "w") as fo:
    json.dump(res, fo, indent=1, ensure_ascii=False)

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
names = {"slices": "срезы одной сцены", "indep": "независимые фото", "realizable": "реализуемые (достижимо 1.0)"}
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
mk = {"M-2 (один проход)": "s", "M-10": "^", "M-10p": "*", "GS+ x20 -> Adam x5": "D", "GS+ x20 -> Adam x10": "D", "GS+ x20 -> Adam x20": "D"}
for j, s in enumerate(sets):
    for i, m in enumerate(["pcc", "ssim"]):
        a = ax[i, j]
        for cname, pts in res["curves"].items():
            a.semilogx([p["ms"] for p in pts], [p[s][m] for p in pts], "o-", label=f"{cname} (K = 1…300)", ms=4)
        for name, p in res["points"].items():
            a.semilogx([p["ms"]], [p[s][m]], mk.get(name, "x"), ms=10 if name == "M-10p" else 7, label=name)
        a.set_xlabel("время синтеза одной маски, мс (MPS)"); a.set_ylabel({"pcc": "КК", "ssim": "SSIM"}[m])
        a.set_title(f"{names[s]}: {'КК' if m == 'pcc' else 'SSIM'}", fontsize=10); a.grid(alpha=.3)
ax[0, 0].legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "out", "e17_quality_vs_time.png"), dpi=110)

for s in sets:
    print(f"\n== {names[s]} ==  (время мс | КК | SSIM)")
    for cname, pts in res["curves"].items():
        for p in pts:
            if p["k"] in (10, 50, 300):
                print(f"  {cname} x{p['k']:<4d}{'':20s} {p['ms']:7.1f} | {p[s]['pcc']:.3f} | {p[s]['ssim']:.3f}")
    for name, p in res["points"].items():
        print(f"  {name:30s} {p['ms']:7.1f} | {p[s]['pcc']:.3f} | {p[s]['ssim']:.3f}")
print("сохранено: out/e17_quality_vs_time.png, out/e17_results.json")
