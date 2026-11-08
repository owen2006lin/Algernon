"""Stage 6: verdict, log-log plot and a markdown summary from check_*.json (+ eval.json if present).

Verdict rules follow the pilot-check doc's "Reading the result" table, with one change found while
testing: the code gate is r(t_min) ~ 1 for the smallest t (0.01), not r(0.1) ~ 1, because a genuine
cubic term can move r(0.1) by tens of percent.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .check import ratios
from .common import load_config, out_path

ORDER = ["step_25", "step_50", "step_75", "final"]


def slope(res, t1, t2):
    k1, k2 = sum(res["kl"][str(t1)]), sum(res["kl"][str(t2)])
    return math.log(k2 / k1) / math.log(t2 / t1)


def verdict(r_new, r_old, tmin_ok):
    if not tmin_ok:
        return "BUG: r(t_min) not close to 1; fix the implementation first"
    if 0.8 <= r_new <= 1.25 and 0.8 <= r_old <= 1.25:
        return "LOCAL: quadratic regime holds at this scale; kappa is meaningful"
    if r_new > 2 or r_old > 2 or r_new < 0.5 or r_old < 0.5:
        if 0.8 <= r_old / r_new <= 1.25:
            return "NON-LOCAL, but r_old/r_new ~ 1: transfer-ratio claims mostly survive (state caveat)"
        return "NON-LOCAL: see Fallbacks"
    if 0.8 <= r_old / r_new <= 1.25:
        return "BORDERLINE: both off, but r_old/r_new ~ 1; transfer-ratio claims mostly survive (state caveat)"
    return "BORDERLINE: r outside [0.8, 1.25] and r_old/r_new differs; treat kappa with caution"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    files = {p.stem.removeprefix("check_"): p for p in Path(cfg["out_dir"]).glob("check_*.json")}
    names = [n for n in ORDER if n in files] + sorted(set(files) - set(ORDER))
    if not names:
        raise SystemExit("no check_*.json found")
    data = {n: json.load(open(files[n])) for n in names}
    lines = ["# Pilot check results", ""]

    ev_path = out_path(cfg, "eval.json")
    if ev_path.exists():
        ev = json.load(open(ev_path))
        lines += ["## Accuracy (base → pilot)", "", "| Benchmark | Base | Pilot | Δ |", "|---|---|---|---|"]
        for b in ("gsm8k", "arc", "mmlu"):
            if b in ev["base"]:
                a0, a1 = ev["base"][b]["acc"], ev["pilot"][b]["acc"]
                lines.append(f"| {b} (n={ev['base'][b]['n']}) | {a0:.3f} | {a1:.3f} | {a1 - a0:+.3f} |")
                if b == "gsm8k":
                    l0, l1 = ev["base"][b]["acc_lenient"], ev["pilot"][b]["acc_lenient"]
                    lines.append(f"| gsm8k, last number | {l0:.3f} | {l1:.3f} | {l1 - l0:+.3f} |")
        lines.append("")

    d0 = data[names[-1] if "final" not in data else "final"]
    ts, e0 = d0["ts"], str(d0["eps"][0])
    tmin = min(ts)

    lines += ["## r(t) = KL / (½ t² dᵀFd), eps = " + e0, "",
              "| Adapter | Set | " + " | ".join(f"r({t})" for t in ts) + " | slope 0.5→1 | KL/tok at t=1 | (r(0.1)−1)/0.1 | eps-sensitivity |",
              "|---|---|" + "---|" * (len(ts) + 4)]
    summary = {}
    for n in names:
        d = data[n]
        for s in ("new", "old"):
            res = d["sets"][s]
            rr = res["ratios"][e0]
            e_alt = res["ratios"][str(d["eps"][-1])]
            eps_sens = max(abs(e_alt[str(t)]["r"] / rr[str(t)]["r"] - 1) for t in ts)
            cubic = (rr["0.1"]["r"] - 1) / 0.1 if "0.1" in rr else float("nan")
            cells = [f"{rr[str(t)]['r']:.3f} [{rr[str(t)]['lo']:.2f}, {rr[str(t)]['hi']:.2f}]" for t in ts]
            lines.append(f"| {n} | {s} | " + " | ".join(cells)
                         + f" | {slope(res, 0.5, 1.0):.2f} | {rr['1.0']['kl_per_tok']:.4f} | {cubic:+.2f} | {eps_sens:.1e} |")
            summary[(n, s)] = rr
    lines.append("")

    lines += ["## Old set by source (final adapter, eps = " + e0 + ")", "",
              "| Source | n seqs | " + " | ".join(f"r({t})" for t in ts) + " |", "|---|---|" + "---|" * len(ts)]
    res_old = d0["sets"]["old"]
    for src in sorted(set(res_old["source"])):
        sel = [i for i, x in enumerate(res_old["source"]) if x == src]
        rr = ratios(res_old, ts, float(e0), sel)
        lines.append(f"| {src} | {len(sel)} | " + " | ".join(f"{rr[str(t)]['r']:.3f}" for t in ts) + " |")
    lines.append("")

    lines += ["## Verdicts", ""]
    for n in names:
        rn, ro = summary[(n, "new")], summary[(n, "old")]
        ok = all(abs(x[str(tmin)]["r"] - 1) < 0.1 for x in (rn, ro))
        lines.append(f"- **{n}**: r_new(1) = {rn['1.0']['r']:.3f}, r_old(1) = {ro['1.0']['r']:.3f}, "
                     f"r_old/r_new = {ro['1.0']['r'] / rn['1.0']['r']:.3f} → {verdict(rn['1.0']['r'], ro['1.0']['r'], ok)}")
    lines.append("")

    # row for the doc's Results table
    lines += ["## Row for the doc's Results table (final adapter)", "",
              "| Run | Prompt set | r(0.1) | r(0.25) | r(0.5) | r(1) | Slope at t = 1 | Verdict |", "|---|---|---|---|---|---|---|---|"]
    for s, label in (("old", "Old"), ("new", "New")):
        rr, res = summary[(names[-1] if "final" not in data else "final", s)], d0["sets"][s]
        lines.append(f"| Pilot SFT | {label} | " + " | ".join(f"{rr[t]['r']:.2f}" for t in ("0.1", "0.25", "0.5", "1.0"))
                     + f" | {slope(res, 0.5, 1.0):.2f} | |")
    text = "\n".join(lines) + "\n"
    out_path(cfg, "summary.md").write_text(text)
    print(text)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, s in zip(axes, ("new", "old")):
            for n in names:
                res = data[n]["sets"][s]
                tok = sum(res["n_tok"])
                kl = [sum(res["kl"][str(t)]) / tok for t in ts]
                q = sum(res["quad"][e0]) / tok
                (line,) = ax.loglog(ts, kl, "o-", label=f"{n} measured")
                ax.loglog(ts, [t**2 * q for t in ts], "--", color=line.get_color(), alpha=0.6,
                          label=f"{n} ½t²dᵀFd")
            ax.set_title(f"{s}-task prompts")
            ax.set_xlabel("t (fraction of LoRA update)")
            ax.grid(True, which="both", alpha=0.3)
        axes[0].set_ylabel("KL(π₀‖π_t) per token (nats)")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_path(cfg, "kl_vs_t.png"), dpi=150)
        print(f"plot: {out_path(cfg, 'kl_vs_t.png')}")
    except Exception as exc:  # plotting is optional
        print(f"plot skipped: {exc}")


if __name__ == "__main__":
    main()
