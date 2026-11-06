"""
Общий цикл self-supervised обучения энкодера ДОЭ (одинаковый для всех вариантов, Р9).

    сеть: 4 изображения -> фаза  ->  Френель x4 (фиксирован)  ->  лосс (1-КК)+(1-SSIM) с целями

Примеры:
  python3 -I train.py --name smoke_m1_digits --model m1 --data digits --minutes 3
  python3 -I train.py --name s1_m2 --model m2 --minutes 12
  python3 -I train.py --name s2_m2_L1 --model m2 --minutes 12 --worst 0.5

Валидация всегда на data/val_div2k.pt (32 четвёрки-слоя DIV2K, сеть их не видит), если не задано --val digits.
Результаты: runs/<name>/ (модель, лог), out/<name>_examples.png, строка в out/results.csv.
"""
import os, sys, json, time, math, argparse, csv
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from physics import Fresnel, evaluate, loss_fn, quantize_phase, pick_device
from models import MODELS, n_params
from data import LayerDataset


def load_digits(split, limit=None):
    from PIL import Image
    root = os.path.join(ROOT, "digits_composite_128_quartets", split)
    ids = sorted(d for d in os.listdir(root) if d.startswith(split))[:limit]
    T = np.stack([[np.asarray(Image.open(os.path.join(root, i, f"{z}.png"))) for z in ("z24p1", "z25p3", "z26p5", "z27p7")]
                  for i in ids]).astype(np.float32) / 255
    T = torch.from_numpy(T)
    return T / T.amax((-2, -1), keepdim=True).clamp_min(1e-8)


class TensorQuartets(torch.utils.data.Dataset):
    def __init__(self, T, length):
        self.T, self.length = T, length

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.T[np.random.randint(len(self.T))]


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def validate(model, Tv, fres, bs=16):
    model.eval()
    P, S = [], []
    for i in range(0, len(Tv), bs):
        phi = model(Tv[i:i + bs], fres)
        p, s, _ = evaluate(phi, Tv[i:i + bs], fres)
        P.append(p); S.append(s)
    model.train()
    P, S = torch.cat(P), torch.cat(S)
    return dict(pcc=P.mean().item(), ssim=S.mean().item(), pcc_min=P.amin(1).mean().item(),
                ssim_min=S.amin(1).mean().item(), pcc_planes=P.mean(0).tolist(), ssim_planes=S.mean(0).tolist())


@torch.no_grad()
def latency(model, Tv, dev, reps=20):
    """время синтеза одной маски (батч 1), медиана, после прогрева"""
    m = model.to(dev).eval(); f = Fresnel().to(dev); x = Tv[:1].to(dev)
    for _ in range(3):
        m(x, f)
    sync(dev)
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); m(x, f); sync(dev); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="m2", choices=list(MODELS))
    ap.add_argument("--data", default="coco", help="coco | digits")
    ap.add_argument("--mode", default="slice", help="slice (D-L1) | independent (D-A)")
    ap.add_argument("--val", default="div2k", help="div2k | digits")
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_ssim", type=float, default=1.0)
    ap.add_argument("--worst", type=float, default=0.0)
    ap.add_argument("--charb", type=float, default=0.0)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = pick_device()
    fres = Fresnel().to(dev)
    run_dir = os.path.join(ROOT, "runs", args.name); os.makedirs(run_dir, exist_ok=True)

    # данные
    if args.data == "digits":
        train_ds = TensorQuartets(load_digits("train"), 10 ** 7)
    else:
        train_ds = LayerDataset(os.path.join(ROOT, "data", f"pool_{args.data}.npz"), length=10 ** 7, mode=args.mode)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.bs, num_workers=2, persistent_workers=True)
    Tv = (load_digits("val", 32) if args.val == "digits" else torch.load(os.path.join(ROOT, "data", "val_div2k.pt"))).to(dev)

    model = MODELS[args.model]().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"[{args.name}] модель {args.model}: {n_params(model) / 1e6:.2f}M параметров, устройство {dev}, бюджет {args.minutes} мин", flush=True)

    log = []
    t0 = time.time(); budget = args.minutes * 60; warm = 200
    step = 0; ema = None
    for T in loader:
        el = time.time() - t0
        if el > budget:
            break
        frac = el / budget
        lr = args.lr * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        T = T.to(dev, non_blocking=True)
        phi = model(T, fres)
        loss, p, s = loss_fn(phi, T, fres, w_ssim=args.w_ssim, worst=args.worst, charb=args.charb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        step += 1
        if step % args.eval_every == 0:
            v = validate(model, Tv, fres)
            log.append(dict(step=step, time_s=round(el, 1), train_loss=ema, **{k: v[k] for k in ("pcc", "ssim", "pcc_min", "ssim_min")}))
            print(f"  шаг {step:6d}  {el / 60:5.1f} мин  loss {ema:.4f}  | вал КК {v['pcc']:.3f} (мин {v['pcc_min']:.3f})  SSIM {v['ssim']:.3f} (мин {v['ssim_min']:.3f})", flush=True)

    v = validate(model, Tv, fres)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    lat = {d: latency(model, Tv, d) for d in ([dev, "cpu"] if dev != "cpu" else ["cpu"])}
    model.to(dev)
    res = dict(name=args.name, model=args.model, params_M=round(n_params(model) / 1e6, 3), data=args.data, mode=args.mode,
               val=args.val, minutes=args.minutes, steps=step, bs=args.bs, lr=args.lr, w_ssim=args.w_ssim,
               worst=args.worst, charb=args.charb, **v, latency_ms={d: round(t * 1e3, 2) for d, t in lat.items()})
    with open(os.path.join(run_dir, "result.json"), "w") as f:
        json.dump(dict(result=res, log=log), f, indent=1, ensure_ascii=False)

    # общая таблица результатов
    csv_path = os.path.join(ROOT, "out", "results.csv")
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "model", "params_M", "data", "mode", "val", "minutes", "steps", "w_ssim", "worst", "charb",
                        "pcc", "pcc_min", "ssim", "ssim_min", "latency_ms_" + dev, "latency_ms_cpu"])
        w.writerow([args.name, args.model, res["params_M"], args.data, args.mode, args.val, args.minutes, step, args.w_ssim,
                    args.worst, args.charb, f"{v['pcc']:.4f}", f"{v['pcc_min']:.4f}", f"{v['ssim']:.4f}",
                    f"{v['ssim_min']:.4f}", res["latency_ms"].get(dev), res["latency_ms"].get("cpu")])

    # картинки: 2 четвёрки, цель / восстановление, + маска
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    with torch.no_grad():
        model.eval(); phi = model(Tv[:2], fres); _, _, rec = evaluate(phi, Tv[:2], fres)
    k, _ = quantize_phase(phi)
    fig, ax = plt.subplots(4, 5, figsize=(11, 9))
    for b in range(2):
        for j in range(4):
            ax[2 * b, j].imshow(Tv[b, j].cpu(), cmap="gray", vmin=0, vmax=1); ax[2 * b, j].set_title(f"цель {j}", fontsize=7)
            ax[2 * b + 1, j].imshow(rec[b, j].cpu(), cmap="gray", vmin=0, vmax=1); ax[2 * b + 1, j].set_title(f"{args.name} {j}", fontsize=7)
        ax[2 * b, 4].imshow(k[b].cpu(), cmap="twilight"); ax[2 * b, 4].set_title("ДОЭ (8 бит)", fontsize=7)
        ax[2 * b + 1, 4].axis("off")
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(os.path.join(ROOT, "out", f"{args.name}_examples.png"), dpi=80)

    print(f"[{args.name}] ИТОГ: шагов {step}, вал КК {v['pcc']:.3f} (мин {v['pcc_min']:.3f}), SSIM {v['ssim']:.3f} (мин {v['ssim_min']:.3f}), "
          f"время маски: " + ", ".join(f"{d} {t * 1e3:.1f} мс" for d, t in lat.items()), flush=True)


if __name__ == "__main__":
    main()
