"""Stage 5: compare KL(pi_0 || pi_t) along the LoRA line with the quadratic prediction.

For each fixed pi_0 rollout and each response position, with z_t the logits at scale t:
  KL_t   = sum_v p_v (log p_v - log p_t,v),   computed stably as log E_p[exp(d - E_p d)], d = z_t - z_0
  quad   = 1/2 Var_p(dz),   dz = (z_eps - z_-eps) / (2 eps)       (= 1/2 d^T F d per position)
  r(t)   = sum KL_t / (t^2 * sum quad)
Writes out_dir/check_<adapter>.json with per-sequence sums and the aggregate r(t).
"""
import argparse
import time

import numpy as np
import torch
from peft import PeftModel

from .common import (adapter_dir, device, left_pad, load_config, load_model, load_tokenizer, out_path,
                     position_ids_from_mask, precise_matmul, set_seed, write_json)
from .lora_scale import set_lora_scale, verify_scaling


def make_batches(rolls, tokens_per_batch):
    """Group sequences of similar length; each batch has at most tokens_per_batch padded tokens."""
    order = sorted(range(len(rolls)), key=lambda i: len(rolls[i]["prompt"]) + len(rolls[i]["response"]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(rolls[i]["prompt"]) + len(rolls[i]["response"])
        if cur and max(cur_max, L) * (len(cur) + 1) > tokens_per_batch:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    return batches


def build_batch(rolls, idx, pad_id, dev):
    seqs = [torch.cat([rolls[i]["prompt"], rolls[i]["response"]]) for i in idx]
    ids, attn, offsets = left_pad(seqs, pad_id)
    rows, cols, owner = [], [], []
    for k, i in enumerate(idx):
        p, r = len(rolls[i]["prompt"]), len(rolls[i]["response"])
        start = offsets[k] + p - 1  # logits at position j predict token j+1
        rows += [k] * r
        cols += list(range(start, start + r))
        owner += [i] * r
    to = lambda x: torch.tensor(x, device=dev)
    return ids.to(dev), attn.to(dev), position_ids_from_mask(attn).to(dev), to(rows), to(cols), to(owner)


@torch.no_grad()
def run_set(model, rolls, cfg, pad_id, dev):
    cc = cfg["check"]
    ts, epss = cc["ts"], cc["eps"]
    n = len(rolls)
    kl = {t: np.zeros(n) for t in ts}
    kl_direct = np.zeros(n)
    quad = {e: np.zeros(n) for e in epss}
    n_tok = np.zeros(n)

    def acc(arr, owner, vals):
        np.add.at(arr, owner.cpu().numpy(), vals.double().cpu().numpy())

    for idx in make_batches(rolls, cc["tokens_per_batch"]):
        ids, attn, pos, rows, cols, owner = build_batch(rolls, idx, pad_id, dev)

        def z(t):
            set_lora_scale(model, t)
            logits = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits
            out = logits[rows, cols].float()
            del logits
            return out

        z0 = z(0.0)
        lp0 = z0.log_softmax(-1)
        P = lp0.exp()
        acc(n_tok, owner, torch.ones_like(rows, dtype=torch.float64))
        for e in epss:
            dz = (z(e) - z(-e)) / (2 * e)
            m = (P * dz).sum(-1, keepdim=True)
            acc(quad[e], owner, 0.5 * (P * (dz - m) ** 2).sum(-1))
            del dz
        for t in ts:
            zt = z(t)
            d = zt - z0
            c = (P * d).sum(-1, keepdim=True)
            acc(kl[t], owner, torch.log1p((P * torch.expm1(d - c)).sum(-1)))
            if t == max(ts):  # naive formula, as a cross-check of the stable one
                acc(kl_direct, owner, (P * (lp0 - zt.log_softmax(-1))).sum(-1))
            del zt, d
        del z0, lp0, P
    set_lora_scale(model, 1.0)
    return {
        "id": [r["id"] for r in rolls],
        "source": [r["source"] for r in rolls],
        "n_tok": n_tok.tolist(),
        "kl": {str(t): kl[t].tolist() for t in ts},
        "kl_direct_tmax": kl_direct.tolist(),
        "quad": {str(e): quad[e].tolist() for e in epss},
    }


def ratios(res, ts, eps, sel=None, n_boot=1000, seed=0):
    """r(t) = sum KL_t / (t^2 sum quad) over the selected sequences, with a paired bootstrap 95% CI."""
    sel = np.arange(len(res["id"])) if sel is None else np.asarray(sel)
    q = np.asarray(res["quad"][str(eps)])[sel]
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(sel), size=(n_boot, len(sel)))
    out = {}
    for t in ts:
        k = np.asarray(res["kl"][str(t)])[sel]
        r = k.sum() / (t**2 * q.sum())
        rb = k[boot].sum(1) / (t**2 * q[boot].sum(1))
        out[str(t)] = {"r": float(r), "lo": float(np.percentile(rb, 2.5)), "hi": float(np.percentile(rb, 97.5)),
                       "kl_per_tok": float(k.sum() / np.asarray(res["n_tok"])[sel].sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapters", nargs="*", help="override check.adapters")
    args = ap.parse_args()
    cfg = load_config(args.config)
    precise_matmul()
    set_seed(cfg["seed"])
    dev = device()
    tok = load_tokenizer(cfg["model_id"])
    rolls = {name: torch.load(out_path(cfg, f"rollouts_{name}.pt")) for name in ("new", "old")}

    base = load_model(cfg["model_id"], dtype=torch.float32, dev=dev)
    for name in args.adapters or cfg["check"]["adapters"]:
        path = adapter_dir(cfg, name)
        if not path.exists():
            print(f"skip {name}: {path} missing")
            continue
        t0 = time.time()
        model = PeftModel.from_pretrained(base, path, is_trainable=False).to(dev).eval()

        # gate: the line parametrisation must be exact before anything is read
        ids, attn, pos, *_ = build_batch(rolls["new"], list(range(min(2, len(rolls["new"])))), tok.pad_token_id, dev)
        ver = verify_scaling(model, ids, attn, pos)
        print(f"[{name}] verify: {ver}")
        if ver["t0_vs_off"] > 1e-4 or ver["t1_vs_merged"] > 1e-2 or ver["t1_vs_t0"] < 1e-4:
            raise SystemExit(f"[{name}] LoRA scaling check failed: {ver}")

        out = {"adapter": name, "verify": ver, "ts": cfg["check"]["ts"], "eps": cfg["check"]["eps"], "sets": {}}
        for set_name, rs in rolls.items():
            res = run_set(model, rs, cfg, tok.pad_token_id, dev)
            res["ratios"] = {str(e): ratios(res, out["ts"], e) for e in out["eps"]}
            out["sets"][set_name] = res
            e0 = str(out["eps"][0])
            print(f"[{name}] {set_name}: " + "  ".join(
                f"r({t})={res['ratios'][e0][str(t)]['r']:.3f}" for t in out["ts"]), flush=True)
        out["seconds"] = time.time() - t0
        write_json(out_path(cfg, f"check_{name}.json"), out)
        base = model.unload()  # strip LoRA modules so the next adapter loads onto a clean base
        print(f"[{name}] done in {out['seconds']:.0f}s")


if __name__ == "__main__":
    main()
