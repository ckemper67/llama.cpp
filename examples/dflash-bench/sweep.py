#!/usr/bin/env python3
# Talk to a running llama-server (loaded ONCE) and sweep the dflash draft length
# per request via the "speculative.n_max" field. No model reloads.
#
# Reads decode t/s and draft acceptance straight from the /completion response.
# Usage: python3 sweep.py [--n-predict 128] [--repeats 3] [--host 127.0.0.1:8080]

import argparse, json, statistics, sys, urllib.request

PROMPT = ("You are a senior systems engineer. Write a detailed, technical explanation of how "
          "speculative decoding accelerates large language model inference on Apple Silicon. "
          "Cover the roles of the target and draft models, how draft tokens are proposed and "
          "verified in a single batched forward pass, what determines the acceptance rate, and "
          "the tradeoffs between draft size, acceptance, and wasted compute.")

SCENARIOS = {
    "greedy":    {"temperature": 0.0, "top_k": 1},
    "mid(t0.7)": {"temperature": 0.7, "top_k": 20, "top_p": 0.95, "min_p": 0.0, "presence_penalty": 0.0},
    # gemma real-workload sampling (models.ini: temp 1.0, top-k 64, top-p 0.95, no penalty)
    "real(t1.0)":{"temperature": 1.0, "top_k": 64, "top_p": 0.95},
}
N_MAX_GRID = [1, 2, 4, 6, 8, 12, 15]
P_MIN_GRID = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]

def call(host, n_max, sampling, n_predict, p_min=None):
    body = {"prompt": PROMPT, "n_predict": n_predict, "seed": 1234,
            "cache_prompt": False, "speculative.n_max": n_max}
    if p_min is not None:
        body["speculative.p_min"] = p_min
    body.update(sampling)
    req = urllib.request.Request("http://%s/completion" % host,
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = None
    for attempt in range(4):  # some reasoning-format models rarely 500 on the content parser
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                resp = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if attempt == 3:
                raise

    t = resp.get("timings", resp)
    tps       = t.get("predicted_per_second", 0.0)
    pred_n    = t.get("predicted_n", n_predict)
    draft_n   = t.get("draft_n", 0)
    draft_acc = t.get("draft_n_accepted", 0)
    accept    = (100.0 * draft_acc / draft_n) if draft_n else 0.0
    # mean accepted tokens per verify cycle: cycles ~= draft_n / n_max
    cycles    = (draft_n / n_max) if (draft_n and n_max) else pred_n
    mean_len  = (pred_n / cycles) if cycles else 1.0
    return tps, accept, mean_len

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--repeats",   type=int, default=3)
    ap.add_argument("--host",      default="127.0.0.1:8080")
    ap.add_argument("--only",      default=None, help="run only this scenario name")
    ap.add_argument("--mode",      default="nmax", choices=["nmax", "pmin"],
                    help="sweep n_max (fixed p_min) or sweep p_min at fixed n_max=15")
    ap.add_argument("--n-max",     type=int, default=15, help="fixed n_max for --mode pmin")
    args = ap.parse_args()

    scenarios = SCENARIOS if not args.only else {args.only: SCENARIOS[args.only]}

    # warm-up (non-fatal: some reasoning-format models 500 on very short generations)
    try:
        call(args.host, 4, SCENARIOS["greedy"], args.n_predict)
    except Exception as e:
        print("warn: warm-up failed (%s); continuing" % e)

    pmin_mode = (args.mode == "pmin")
    axis_vals = P_MIN_GRID if pmin_mode else N_MAX_GRID
    axis_lbl  = "p_min" if pmin_mode else "n_max"

    best = {}
    for name, samp in scenarios.items():
        print("\n--- scenario: %s (%s sweep%s) ---"
              % (name, axis_lbl, (", n_max=%d" % args.n_max) if pmin_mode else ""))
        print("%-7s %-9s %-9s %-9s" % (axis_lbl, "t/s", "accept%", "mean_len"))
        best[name] = (None, -1.0, 0.0, 0.0)
        for v in axis_vals:
            nm    = args.n_max if pmin_mode else v
            p_min = v         if pmin_mode else None
            tps_r, acc_r, ml_r = [], [], []
            for _ in range(args.repeats):
                tps, acc, ml = call(args.host, nm, samp, args.n_predict, p_min)
                tps_r.append(tps); acc_r.append(acc); ml_r.append(ml)
            tps = statistics.median(tps_r); acc = statistics.median(acc_r); ml = statistics.median(ml_r)
            print(("%-7.2f " if pmin_mode else "%-7d ") % v + "%-9.2f %-9.2f %-9.2f" % (tps, acc, ml))
            if tps > best[name][1]:
                best[name] = (v, tps, acc, ml)

    print("\n=== recommended (per scenario) ===")
    for name, (v, tps, acc, ml) in best.items():
        setting = ("n_max=15 spec-draft-p-min = %.2f" % v) if pmin_mode else ("spec-draft-n-max = %d" % v)
        print("%-11s : %s  (%.1f t/s, accept %.1f%%, mean_len %.2f)" % (name, setting, tps, acc, ml))

if __name__ == "__main__":
    main()
