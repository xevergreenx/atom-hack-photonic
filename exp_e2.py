"""
Э2 (этап 0a): цели для нейросети на слоях сцены.

- Фиксированный валидационный набор: 32 четвёрки-слоя из DIV2K (seed 123, без аугментаций),
  сохраняется в data/val_div2k.pt и используется всеми экспериментами (Р9).
- Суперпозиция, многоплоскостной GS (кривая качества по итерациям), Adam по фазе (потолок).
- Время синтеза одной маски (батч 1) на CPU и MPS.

Запуск: python3 -I exp_e2.py
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from physics import Fresnel, evaluate, loss_fn, superposition, gs, pick_device
from data import LayerDataset

torch.manual_seed(0)
DEV = pick_device()
ROOT = os.path.dirname(os.path.abspath(__file__))
VAL_PATH = os.path.join(ROOT, "data", "val_div2k.pt")
CKPT = [1, 2, 5, 10, 20, 50, 100, 200, 300]


def make_val(n=32):
    if os.path.exists(VAL_PATH):
        return torch.load(VAL_PATH)
    ds = LayerDataset(os.path.join(ROOT, "data", "pool_div2k.npz"), length=n, mode="slice", aug=False, seed=123)
    T = torch.stack([ds[i] for i in range(n)])
    torch.save(T, VAL_PATH)
    return T


def summarize(p, s):
    return dict(pcc=p.mean().item(), ssim=s.mean().item(),
                pcc_min=p.amin(1).mean().item(), ssim_min=s.amin(1).mean().item(),
                pcc_planes=p.mean(0).tolist(), ssim_planes=s.mean(0).tolist())


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def time_gs(T1, dev, iters, reps=3):
    fres = Fresnel().to(dev)
    T1 = T1.to(dev)
    gs(T1, fres, iters=2); sync(dev)                           # прогрев
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); gs(T1, fres, iters=iters); sync(dev); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    T = make_val().to(DEV)
    fres = Fresnel().to(DEV)
    res = {"device": DEV, "n_val": len(T)}

    # суперпозиция
    phi = superposition(T, fres)
    res["superposition"] = summarize(*evaluate(phi, T, fres)[:2])

    # GS: качество по итерациям
    curve = {}
    def cb(it, phi):
        if it in CKPT:
            curve[it] = summarize(*evaluate(phi, T, fres)[:2])
    phi_gs = gs(T, fres, iters=max(CKPT), callback=cb)
    res["gs_curve"] = curve

    # GS: время одной маски
    tm = {}
    for dev in ["cpu", DEV] if DEV != "cpu" else ["cpu"]:
        tm[dev] = {k: time_gs(T[:1], dev, k) for k in CKPT}
    res["gs_time_s"] = tm

    # Adam по фазе (потолок; в сабмит не идёт): батчем, качество на контрольных точках
    adam = {}
    p = superposition(T, fres).clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=0.05)
    for it in range(1, 1001):
        opt.zero_grad()
        loss, _, _ = loss_fn(p, T, fres)
        loss.backward()
        opt.step()
        if it in (100, 300, 1000):
            adam[it] = summarize(*evaluate(p.detach(), T, fres)[:2])
    res["adam"] = adam
    # время Adam для одной маски (1000 шагов)
    q = superposition(T[:1], fres).clone().requires_grad_(True); o = torch.optim.Adam([q], lr=0.05)
    sync(DEV); t = time.perf_counter()
    for _ in range(1000):
        o.zero_grad(); l, _, _ = loss_fn(q, T[:1], fres); l.backward(); o.step()
    sync(DEV); res["adam_time_s_1000"] = time.perf_counter() - t

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    with open(os.path.join(ROOT, "out", "e2_results.json"), "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)

    # таблица
    print(f"\nЭ2: {len(T)} четвёрок-слоёв DIV2K, устройство {DEV}")
    print(f"{'метод':22s} | {'КК ср':>6s} {'КК мин':>6s} | {'SSIM ср':>7s} {'SSIM мин':>8s} | время CPU / {DEV}")
    r = res["superposition"]
    print(f"{'суперпозиция':22s} | {r['pcc']:6.3f} {r['pcc_min']:6.3f} | {r['ssim']:7.3f} {r['ssim_min']:8.3f} |")
    for k in CKPT:
        r = curve[k]
        tt = " / ".join(f"{tm[d][k]*1e3:7.1f} мс" for d in tm)
        print(f"{'GS x' + str(k):22s} | {r['pcc']:6.3f} {r['pcc_min']:6.3f} | {r['ssim']:7.3f} {r['ssim_min']:8.3f} | {tt}")
    for k, r in adam.items():
        print(f"{'Adam x' + str(k):22s} | {r['pcc']:6.3f} {r['pcc_min']:6.3f} | {r['ssim']:7.3f} {r['ssim_min']:8.3f} |")
    print(f"Adam x1000, одна маска: {res['adam_time_s_1000']:.1f} с на {DEV}")

    # график качество-время
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d in tm:
        xs = [tm[d][k] * 1e3 for k in CKPT]
        ax[0].semilogx(xs, [curve[k]["pcc"] for k in CKPT], "o-", label=f"GS ({d})")
        ax[1].semilogx(xs, [curve[k]["ssim"] for k in CKPT], "o-", label=f"GS ({d})")
    for a, m in zip(ax, ["pcc", "ssim"]):
        a.axhline(adam[1000][m], ls="--", c="gray", label="Adam x1000 (потолок)")
        a.set_xlabel("время синтеза одной маски, мс"); a.set_ylabel({"pcc": "КК", "ssim": "SSIM"}[m]); a.grid(alpha=.3); a.legend(fontsize=8)
    ax[0].set_title("Э2: GS на слоях DIV2K — качество от времени")
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "out", "e2_gs_curve.png"), dpi=110)

    # примеры восстановлений
    _, _, rec_gs = evaluate(phi_gs, T, fres)
    _, _, rec_ad = evaluate(p.detach(), T, fres)
    fig, ax = plt.subplots(6, 4, figsize=(8, 12))
    for b in range(2):
        for j in range(4):
            for r_, (img, name) in enumerate([(T, "цель"), (rec_gs, "GS x300"), (rec_ad, "Adam x1000")]):
                a = ax[b * 3 + r_, j]; a.imshow(img[b, j].cpu(), cmap="gray", vmin=0, vmax=1); a.axis("off")
                a.set_title(f"{name}, плоскость {j}", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "out", "e2_examples.png"), dpi=90)
    print("сохранено: out/e2_results.json, out/e2_gs_curve.png, out/e2_examples.png")


if __name__ == "__main__":
    main()
