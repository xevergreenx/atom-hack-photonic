"""
Часть II журнала: «В случае с дотюнингом». Допущение: под конкретную четвёрку МОЖНО оптимизировать —
шаги по фазе и/или веса сети. Скорость в оценке не участвует (слайд 6 ТЗ), поэтому бюджет — секунды на маску.

Д-1. Прямая оптимизация фазы по метрикам (Adam, лосс L-0), длинный бюджет, три старта:
       super  — суперпозиция со случайной фазой (как в Э12b);
       gsp    — ядро «GS + веса + связь» x300 (лучшее классическое ядро, Э15);
       m2     — выход предобученной M-2 (один проход сети).
Д-2. Нейросетевая оптимизация под четвёрку: оптимизируются ВЕСА сети, а не фаза напрямую.
       dip    — сеть с нуля (как PhysenNet / Deep Image Prior): U-Net(цели) -> (cos, sin) -> фаза;
       res    — сеть-поправка поверх gsp: фаза = arg(e^{iφ_gsp} + c(цели, φ_gsp)), выход сети на старте 0;
       m2ft   — предобученная M-2 + дообучение её весов под четвёрку.
     У каждой четвёрки свои независимые веса (torch.func: stack_module_state + vmap), Adam поэлементный,
     поэтому 32 четвёрки считаются одним батчем, но друг на друга не влияют.
Д-3. Выбор лучшего кандидата по своим же метрикам: цели известны в момент синтеза, метод сам считает
     КК и SSIM у всех кандидатов и отдаёт лучший (по сумме КК + SSIM).

Оценка у всех — как у жюри (physics.evaluate): фаза -> 8 бит -> Френель -> амплитуда/max -> 8 бит -> КК, SSIM.

Запуск (как в журнале, часть II):
  этап A, фаза, 32 четвёрки:  python3 -I exp_d1.py --methods super,gsp,m2 --out d1a_results.json
  этап B, сети, 8 четвёрок:   python3 -I exp_d1.py --n 8 --methods gsp,dip,res,m2ft --out d1b_results.json
  проверка, что всё работает: python3 -I exp_d1.py --n 4 --steps 20 --smoke
"""
import os, sys, json, time, copy, argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import torch
from torch.func import stack_module_state, functional_call, vmap
from physics import Fresnel, evaluate, ssim_sk, loss_fn, superposition, gs_plus, pick_device
from models import UNet, M2

ap = argparse.ArgumentParser()
ap.add_argument("--device", default=None)
ap.add_argument("--sets", default="slices,indep,realizable")
ap.add_argument("--n", type=int, default=32, help="четвёрок из каждого набора")
ap.add_argument("--steps", type=int, default=1000, help="шагов оптимизации у каждого метода")
ap.add_argument("--methods", default="super,gsp,m2,dip,res,m2ft")
ap.add_argument("--lr_phase", type=float, default=0.05)
ap.add_argument("--lr_dip", type=float, default=3e-3)
ap.add_argument("--lr_res", type=float, default=3e-3)
ap.add_argument("--lr_m2ft", type=float, default=1e-3)
ap.add_argument("--smoke", action="store_true", help="не перезаписывать out/d1_results.json")
ap.add_argument("--out", default="d1_results.json", help="файл результатов в out/")
args = ap.parse_args()

DEV = args.device or pick_device()
f = Fresnel().to(DEV)
CKPTS = sorted({0, 100, 300, 1000, 2000, args.steps} & set(range(args.steps + 1)))
SET_FILES = {"slices": "val_div2k.pt", "indep": "val_div2k_indep.pt", "realizable": "val_realizable.pt"}


def sync():
    if DEV == "mps":
        torch.mps.synchronize()
    elif DEV == "cuda":
        torch.cuda.synchronize()


def score(phi, T):
    """метрики по четвёркам: КК (B,4), SSIM (B,4), SSIM-sk (B,4)"""
    p, s, rec = evaluate(phi.detach(), T, f)
    return p, s, ssim_sk(rec, T)


def summary(p, s, s2):
    r = lambda x: round(x.item(), 4)
    return dict(pcc=r(p.mean()), pcc_min=r(p.amin(1).mean()), ssim=r(s.mean()), ssim_sk=r(s2.mean()),
                ssim_min=r(s.amin(1).mean()))


def optimize(params, phase_fn, T, lr):
    """Adam по L-0; params — список листовых тензоров. Возвращает кривую {K: метрики}, финальную фазу, время."""
    opt = torch.optim.Adam(params, lr=lr)
    curve, t_opt = {}, 0.0
    with torch.no_grad():
        curve[0] = score(phase_fn(), T)
    for it in range(1, args.steps + 1):
        sync(); t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, _, _ = loss_fn(phase_fn(), T, f)
        loss.backward()
        opt.step()
        sync(); t_opt += time.perf_counter() - t0
        if it in CKPTS:
            with torch.no_grad():
                curve[it] = score(phase_fn(), T)
    with torch.no_grad():
        phi = phase_fn().detach()
    return curve, phi, t_opt


class PerQuartet:
    """Независимые копии сети для каждой четвёрки: веса сложены по оси 0, прямой проход — vmap."""

    def __init__(self, models):
        self.base = copy.deepcopy(models[0]).to("meta")
        self.params, self.buffers = stack_module_state(models)
        self.params = {k: v.detach().to(DEV).requires_grad_(True) for k, v in self.params.items()}
        self.buffers = {k: v.to(DEV) for k, v in self.buffers.items()}

    def __call__(self, fn, *xs):
        """fn(module_call, *x_one) -> выход одной четвёрки; xs батчированы по оси 0"""
        def one(p, b, *x):
            call = lambda *a: functional_call(self.base, (p, b), a)
            return fn(call, *x)
        return vmap(one)(self.params, self.buffers, *xs)


def load_m2():
    for name in ("e11_m2_long", "s1_m2"):
        path = os.path.join(ROOT, "runs", name, "model.pt")
        if os.path.exists(os.path.join(ROOT, "runs", name, "result.json")):
            m = M2().to(DEV); m.load_state_dict(torch.load(path, map_location=DEV)); m.eval()
            return m, name
    raise FileNotFoundError("нет обученной M-2")


m2, m2_name = load_m2()
print(f"устройство {DEV}, M-2 из runs/{m2_name}, шагов {args.steps}, контрольные точки {CKPTS}", flush=True)

methods = args.methods.split(",")
res, finals = {}, {}
for sname in args.sets.split(","):
    T = torch.load(os.path.join(ROOT, "data", SET_FILES[sname]))[: args.n].to(DEV)
    B = T.shape[0]
    torch.manual_seed(1)
    phi_sup = superposition(T, f)
    with torch.no_grad():
        sync(); t0 = time.perf_counter()
        phi_gsp = gs_plus(T, f, iters=300, phi0=phi_sup)
        sync(); t_gsp = time.perf_counter() - t0
        phi_m2 = m2(T, f)
    res[sname], finals[sname] = {}, {"gsp x300": score(phi_gsp, T)}
    res[sname]["gsp x300 (без дотюнинга)"] = {"curve": {0: summary(*finals[sname]["gsp x300"])},
                                              "sec_per_mask": round(t_gsp / B, 4)}

    for mname in methods:
        torch.manual_seed(2)
        if mname in ("super", "gsp", "m2"):                       # Д-1: Adam прямо по фазе
            start = {"super": phi_sup, "gsp": phi_gsp, "m2": phi_m2}[mname]
            p = start.clone().requires_grad_(True)
            curve, phi, t = optimize([p], lambda: p, T, args.lr_phase)
        elif mname == "dip":                                      # Д-2a: сеть с нуля
            pq = PerQuartet([UNet(4, 2, base=16) for _ in range(B)])
            def phase_fn():
                o = pq(lambda call, t: call(t[None])[0], T)       # (B,2,N,N)
                return torch.atan2(o[:, 1], o[:, 0])
            curve, phi, t = optimize(list(pq.params.values()), phase_fn, T, args.lr_dip)
        elif mname == "res":                                      # Д-2b: сеть-поправка поверх gsp
            pq = PerQuartet([UNet(6, 2, base=16, zero_out=True) for _ in range(B)])
            c0, s0 = torch.cos(phi_gsp), torch.sin(phi_gsp)
            x = torch.cat([T, c0[:, None], s0[:, None]], 1)
            def phase_fn():
                o = pq(lambda call, xx: call(xx[None])[0], x)
                return torch.atan2(s0 + o[:, 1], c0 + o[:, 0])
            curve, phi, t = optimize(list(pq.params.values()), phase_fn, T, args.lr_res)
        elif mname == "m2ft":                                     # Д-2c: предобученная M-2 + дообучение весов
            pq = PerQuartet([copy.deepcopy(m2) for _ in range(B)])
            phase_fn = lambda: pq(lambda call, t: call(t[None], f)[0], T)
            curve, phi, t = optimize(list(pq.params.values()), phase_fn, T, args.lr_m2ft)
        else:
            raise ValueError(mname)
        label = {"super": "Д-1 super -> Adam", "gsp": "Д-1 gsp x300 -> Adam", "m2": "Д-1 M-2 -> Adam",
                 "dip": "Д-2a сеть с нуля (DIP)", "res": "Д-2b сеть-поправка поверх gsp",
                 "m2ft": "Д-2c M-2 + дообучение весов"}[mname]
        res[sname][label] = {"curve": {k: summary(*v) for k, v in curve.items()},
                             "sec_per_mask": round((t + (t_gsp if mname in ("gsp", "res") else 0)) / B, 3)}
        finals[sname][label] = curve[max(curve)]
        last = res[sname][label]["curve"][max(curve)]
        print(f"[{sname}] {label:32s} K={max(curve):5d}: КК {last['pcc']:.3f} (мин {last['pcc_min']:.3f})  "
              f"SSIM {last['ssim']:.3f} / sk {last['ssim_sk']:.3f}   {res[sname][label]['sec_per_mask']:.2f} с/маска",
              flush=True)

    # Д-3: лучший кандидат по своим метрикам (по сумме КК + SSIM, среднее по 4 плоскостям)
    names = list(finals[sname])
    P = torch.stack([finals[sname][k][0] for k in names])         # (M,B,4)
    S = torch.stack([finals[sname][k][1] for k in names])
    S2 = torch.stack([finals[sname][k][2] for k in names])
    pick = (P.mean(-1) + S.mean(-1)).argmax(0)                    # (B,)
    ar = torch.arange(B, device=P.device)
    res[sname]["Д-3 лучший кандидат"] = {"curve": {0: summary(P[pick, ar], S[pick, ar], S2[pick, ar])},
                                         "picked": {k: int((pick == i).sum()) for i, k in enumerate(names)}}

out_path = os.path.join(ROOT, "out", "d1_smoke.json" if args.smoke else args.out)
with open(out_path, "w") as fo:
    json.dump(dict(device=DEV, m2=m2_name, steps=args.steps, n=args.n, lr=dict(phase=args.lr_phase, dip=args.lr_dip,
                   res=args.lr_res, m2ft=args.lr_m2ft), results=res), fo, indent=1, ensure_ascii=False)

for sname, r in res.items():
    print(f"\n== {sname} ==  КК / КК мин / SSIM / SSIM-sk по K")
    for label, v in r.items():
        cells = "  ".join(f"K={k}: {c['pcc']:.3f}/{c['pcc_min']:.3f}/{c['ssim']:.3f}/{c['ssim_sk']:.3f}"
                          for k, c in v["curve"].items())
        print(f"{label:34s} {cells}")
    print("Д-3, сколько раз выбран каждый кандидат:", r["Д-3 лучший кандидат"]["picked"])
print("\nсохранено:", out_path)
