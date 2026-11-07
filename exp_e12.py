"""
Э12 (Р13): страховка — все методы на трёх валидационных наборах, SSIM в двух реализациях.

Наборы (по 32 четвёрки, seed 123, без аугментаций):
  slices      — срезы одной сцены DIV2K по глубине (data/val_div2k.pt, как раньше)
  indep       — 4 независимых фото DIV2K (data/val_div2k_indep.pt)
  realizable  — реализуемые четвёрки: маска GS x50 по independent -> 8 бит -> 4 восстановления Френелем.
                Для них существует маска с КК = 1 (data/val_realizable.pt)

Методы: суперпозиция, GS x50, GS x300, гибрид GS x20 -> 10 шагов Adam, M-2 (один проход), M-10, M-2 long (если обучена).
Запуск (на CPU, чтобы не мешать обучению на GPU): python3 -I exp_e12.py --device cpu
"""
import os, sys, json, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from physics import Fresnel, evaluate, ssim_sk, quantize_phase, quantize_img, to_image, loss_fn, superposition, gs, gs_plus
from models import MODELS
from data import LayerDataset

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
args = ap.parse_args()
DEV = args.device
torch.manual_seed(0)
f = Fresnel().to(DEV)


def make_sets():
    p_sl = os.path.join(ROOT, "data", "val_div2k.pt")
    p_in = os.path.join(ROOT, "data", "val_div2k_indep.pt")
    p_re = os.path.join(ROOT, "data", "val_realizable.pt")
    if not os.path.exists(p_in):
        ds = LayerDataset(os.path.join(ROOT, "data", "pool_div2k.npz"), length=32, mode="independent", aug=False, seed=123)
        torch.save(torch.stack([ds[i] for i in range(32)]), p_in)
    if not os.path.exists(p_re):
        Ti = torch.load(p_in).to(DEV)
        with torch.no_grad():
            phi = gs(Ti, f, iters=50)
            _, phq = quantize_phase(phi)
            Tr = quantize_img(to_image(f(phq)))          # ровно то, что даёт 8-битная маска
        torch.save(Tr.cpu(), p_re)
    return {k: torch.load(p).to(DEV) for k, p in [("slices", p_sl), ("indep", p_in), ("realizable", p_re)]}


def hybrid(T, g=20, k=10, lr=0.1):
    with torch.no_grad():
        phi = gs(T, f, iters=g)
    p = phi.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(k):
        opt.zero_grad(); l, _, _ = loss_fn(p, T, f); l.backward(); opt.step()
    return p.detach()


def load(name, ckpt):
    path = os.path.join(ROOT, "runs", ckpt, "model.pt")
    if not os.path.exists(path):
        return None
    m = MODELS[name]().to(DEV); m.load_state_dict(torch.load(path, map_location=DEV)); m.eval()
    return lambda T: m(T, f).detach()


def hybrid_plus(T, g=20, k=10, lr=0.1):
    with torch.no_grad():
        phi = gs_plus(T, f, iters=g)
    p = phi.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(k):
        opt.zero_grad(); l, _, _ = loss_fn(p, T, f); l.backward(); opt.step()
    return p.detach()


def m10p_untrained(T):
    m = MODELS["m10p"]().to(DEV).eval()                  # поправка сети = 0, α = 0.15: вклад обучения = разница с обученной
    return m(T, f).detach()


methods = {
    "суперпозиция": lambda T: superposition(T, f),
    "GS x50": lambda T: gs(T, f, iters=50),
    "GS x300": lambda T: gs(T, f, iters=300),
    "GS+ x50 (веса + связь)": lambda T: gs_plus(T, f, iters=50),
    "GS+ x300 (веса + связь)": lambda T: gs_plus(T, f, iters=300),
    "гибрид GS x20 -> Adam x10": hybrid,
    "гибрид GS+ x20 -> Adam x10": hybrid_plus,
    "M-2 (один проход, 15 мин)": load("m2", "s1_m2"),
    "M-2 (один проход, 60 мин)": load("m2", "e11_m2_long"),
    "M-10 (GS x20 + 3 обуч. шага, срезы)": load("m10", "s1_m10"),
    "M-10p без обучения": m10p_untrained,
    "M-10p (GS+ x20 + 3 обуч. шага, смесь)": load("m10p", "f1_m10p_mix"),
}
methods = {k: v for k, v in methods.items() if v is not None}

sets = make_sets()
res = {}
for mname, fn in methods.items():
    res[mname] = {}
    for sname, T in sets.items():
        phi = fn(T)
        p, s, rec = evaluate(phi, T, f)
        s2 = ssim_sk(rec, T)
        res[mname][sname] = dict(pcc=round(p.mean().item(), 4), pcc_min=round(p.amin(1).mean().item(), 4),
                                 ssim=round(s.mean().item(), 4), ssim_sk=round(s2.mean().item(), 4),
                                 ssim_min=round(s.amin(1).mean().item(), 4))
    print("готово:", mname, flush=True)

OUT_JSON = os.environ.get("E12_OUT", "e12_results.json")
with open(os.path.join(ROOT, "out", OUT_JSON), "w") as fo:
    json.dump(res, fo, indent=1, ensure_ascii=False)

for sname in sets:
    print(f"\n== набор: {sname} ==")
    print(f"{'метод':40s} | {'КК':>6s} {'КК мин':>6s} | {'SSIM':>6s} {'SSIM-sk':>7s} {'SSIM мин':>8s}")
    for mname in methods:
        r = res[mname][sname]
        print(f"{mname:40s} | {r['pcc']:6.3f} {r['pcc_min']:6.3f} | {r['ssim']:6.3f} {r['ssim_sk']:7.3f} {r['ssim_min']:8.3f}")
