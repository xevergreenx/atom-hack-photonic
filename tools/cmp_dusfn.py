"""Сравнение с DUSFN v3 на их 6 тестовых четвёрках: их маски через наш оценщик + наши методы на тех же целях."""
import sys, os, csv, json
PROJ = "/Users/evergreenpc/Documents/project/atom-hack"
sys.path.insert(0, PROJ)
import numpy as np, torch
from PIL import Image
from physics import Fresnel, evaluate, loss_fn, gs, gs_plus, pick_device
from models import MODELS

D = sys.argv[1]
dev = pick_device()
f = Fresnel().to(dev)
cases = sorted(os.listdir(os.path.join(D, "previews")))

def load_targets(c):
    T = np.stack([np.asarray(Image.open(os.path.join(D, "previews", c, f"target_{j}.png")).convert("L")) for j in range(1, 5)])
    T = torch.from_numpy(T.astype(np.float32) / 255)[None]
    return (T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)).to(dev)

def load_doe(c):
    k = torch.from_numpy(np.asarray(Image.open(os.path.join(D, "previews", c, "doe_phase_uint8.png"))).astype(np.float32))
    return (k / 255 * 2 * np.pi - np.pi)[None].to(dev)

reported = {}
with open(os.path.join(D, "test_per_plane.csv")) as fo:
    for r in csv.DictReader(fo):
        reported.setdefault(r["sample_id"], {"pcc": [], "ssim": [], "family": r["family"], "comp": r["composition"]})
        reported[r["sample_id"]]["pcc"].append(float(r["PCC"])); reported[r["sample_id"]]["ssim"].append(float(r["SSIM_local"]))

m10p = MODELS["m10p"]().to(dev); m10p.load_state_dict(torch.load(os.path.join(PROJ, "runs/f1_m10p_mix/model.pt"), map_location=dev)); m10p.eval()

def adam_after(core, g, k, lr=0.1):
    def run(T):
        with torch.no_grad():
            phi = core(T, f, iters=g) if g else None
        if phi is None:
            from physics import superposition
            phi = superposition(T, f)
        p = phi.clone().requires_grad_(True); o = torch.optim.Adam([p], lr=lr)
        for _ in range(k):
            o.zero_grad(); l, _, _ = loss_fn(p, T, f); l.backward(); o.step()
        return p.detach()
    return run

methods = {
    "DUSFN (их маска, наш оценщик)": None,
    "GS x300": lambda T: gs(T, f, iters=300),
    "GS+ x300": lambda T: gs_plus(T, f, iters=300),
    "M-10p (наш)": lambda T: m10p(T, f).detach(),
    "GS+ x20 -> Adam x10": adam_after(gs_plus, 20, 10),
    "Adam x200 по метрикам (дотюнинг)": adam_after(None, 0, 200, lr=0.05),
}
res = {m: [] for m in methods}
print(f"{'четвёрка':12s} {'семейство / состав':45s} | отчёт DUSFN КК/SSIM | пересчёт нашим кодом")
for c in cases:
    T = load_targets(c)
    torch.manual_seed(0)
    for m, fn in methods.items():
        phi = load_doe(c) if fn is None else fn(T)
        p, s, _ = evaluate(phi, T, f)
        res[m].append((p.mean().item(), s.mean().item(), p.min().item()))
    rp = reported.get(c)
    rep = f"{np.mean(rp['pcc']):.3f} / {np.mean(rp['ssim']):.3f}" if rp else "—"
    ours = res["DUSFN (их маска, наш оценщик)"][-1]
    print(f"{c:12s} {(rp['family'] + ' / ' + rp['comp']) if rp else '':45s} | {rep:19s} | {ours[0]:.3f} / {ours[1]:.3f}")

print(f"\nСреднее по {len(cases)} четвёркам:  КК / SSIM / худшая плоскость КК")
for m, v in res.items():
    a = np.array(v)
    print(f"  {m:36s} {a[:,0].mean():.3f} / {a[:,1].mean():.3f} / {a[:,2].mean():.3f}")
print("\nПо четвёркам (КК / SSIM):")
for i, c in enumerate(cases):
    print(f"  {c}: " + " | ".join(f"{m.split(' (')[0]} {res[m][i][0]:.2f}/{res[m][i][1]:.2f}" for m in methods))
json.dump({m: v for m, v in res.items()}, open(os.path.join(PROJ, "out", "cmp_dusfn.json"), "w"), indent=1, ensure_ascii=False)
