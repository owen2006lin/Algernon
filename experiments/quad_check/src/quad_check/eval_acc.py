"""Stage 3: accuracy of base vs pilot. GSM8K by greedy generation; ARC / MMLU by answer-letter logits.

This guards against a pilot that learned nothing (then KL is tiny and r ~ 1 is meaningless),
and gives the old-benchmark drops. Writes out_dir/eval.json.
"""
import argparse
import contextlib
import re

import torch
from peft import PeftModel
from transformers import GenerationConfig

from .common import (adapter_dir, device, left_pad, load_config, load_model, load_tokenizer, native_bf16, out_path,
                     position_ids_from_mask, prompt_ids, read_jsonl, set_seed, stop_token_ids, write_json)

NUM = re.compile(r"####\s*\$?\s*(-?[\d,]*\.?\d+)")


LAST_NUM = re.compile(r"-?[\d,]*\.?\d+")


def extract(text, lenient=False):
    """Strict: the '#### n' line. Lenient: fall back to the last number in the text."""
    m = NUM.findall(text)
    if not m and lenient:
        m = LAST_NUM.findall(text)
    if not m:
        return None
    s = m[-1].replace(",", "")
    try:
        v = float(s)
    except ValueError:
        return None
    return str(int(v)) if v == int(v) else str(v)


@torch.no_grad()
def gsm8k_acc(model, tok, rows, bs, max_new, dev):
    gen = GenerationConfig(do_sample=False, max_new_tokens=max_new, repetition_penalty=1.0,
                           temperature=None, top_p=None, top_k=None,
                           eos_token_id=stop_token_ids(tok), pad_token_id=tok.pad_token_id)
    correct, lenient, fmt, samples = 0, 0, 0, []
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        ids, attn, _ = left_pad([torch.tensor(prompt_ids(tok, r["messages"])) for r in batch], tok.pad_token_id)
        out = model.generate(input_ids=ids.to(dev), attention_mask=attn.to(dev), generation_config=gen)
        texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
        for r, t in zip(batch, texts):
            pred = extract(t)
            fmt += pred is not None
            correct += pred == r["gold"]
            lenient += extract(t, lenient=True) == r["gold"]
            if len(samples) < 3:
                samples.append({"gold": r["gold"], "pred": pred, "text": t[-300:]})
    return {"acc": correct / len(rows), "acc_lenient": lenient / len(rows), "format_rate": fmt / len(rows),
            "n": len(rows), "samples": samples}


@torch.no_grad()
def mc_acc(model, tok, rows, bs, dev):
    letter_ids = [tok(L, add_special_tokens=False).input_ids[0] for L in "ABCDEFGH"]
    res = {}
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        ids, attn, _ = left_pad([torch.tensor(prompt_ids(tok, r["messages"])) for r in batch], tok.pad_token_id)
        ids, attn = ids.to(dev), attn.to(dev)
        logits = model(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn)).logits[:, -1]
        for r, z in zip(batch, logits):
            k = r["n_choices"]
            pred = "ABCDEFGH"[int(z[letter_ids[:k]].argmax())]
            s = res.setdefault(r["source"], [0, 0])
            s[0] += pred == r["gold"]
            s[1] += 1
    return {src: {"acc": c / n, "n": n} for src, (c, n) in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--adapter", default="final")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ec = cfg["eval"]
    set_seed(cfg["seed"])
    dev = device()
    tok = load_tokenizer(cfg["model_id"])
    dtype = torch.bfloat16 if native_bf16() else torch.float32
    base = load_model(cfg["model_id"], dtype=dtype, dev=dev)
    model = PeftModel.from_pretrained(base, adapter_dir(cfg, args.adapter)).to(dev).eval()
    gsm = read_jsonl(out_path(cfg, "data", "eval_gsm8k.jsonl"))
    mc = read_jsonl(out_path(cfg, "data", "eval_mc.jsonl"))

    result = {"adapter": args.adapter}
    for name, ctx in (("base", model.disable_adapter), ("pilot", contextlib.nullcontext)):
        with ctx():
            g = gsm8k_acc(model, tok, gsm, ec["batch_size"], ec["max_new_tokens"], dev)
            m = mc_acc(model, tok, mc, ec["batch_size"], dev)
        result[name] = {"gsm8k": g, **m}
        print(f"{name}: gsm8k {g['acc']:.3f} (lenient {g['acc_lenient']:.3f}, format {g['format_rate']:.2f})  "
              + "  ".join(f"{k} {v['acc']:.3f}" for k, v in m.items()), flush=True)
    write_json(out_path(cfg, "eval.json"), result)


if __name__ == "__main__":
    main()
