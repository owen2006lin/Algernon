"""Stage 4: sample one response per check prompt from the base model pi_0.

Pure ancestral sampling: T=1, no top-p/top-k, no repetition penalty (Qwen's
generation_config sets all three, so they are overridden explicitly). Saves
out_dir/rollouts_{old,new}.pt: a list of dicts with prompt/response token ids.
"""
import argparse

import torch
from transformers import GenerationConfig

from .common import (device, left_pad, load_config, load_model, load_tokenizer, out_path,
                     precise_matmul, prompt_ids, read_jsonl, set_seed, stop_token_ids)


@torch.no_grad()
def sample_set(model, tok, rows, bs, max_new, dev):
    stops = stop_token_ids(tok)
    gen = GenerationConfig(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, min_p=None,
                           repetition_penalty=1.0, max_new_tokens=max_new,
                           eos_token_id=stops, pad_token_id=tok.pad_token_id)
    out = []
    for i in range(0, len(rows), bs):
        batch = rows[i : i + bs]
        p = [torch.tensor(prompt_ids(tok, r["messages"])) for r in batch]
        ids, attn, _ = left_pad(p, tok.pad_token_id)
        seq = model.generate(input_ids=ids.to(dev), attention_mask=attn.to(dev), generation_config=gen).cpu()
        resp = seq[:, ids.shape[1]:]
        for r, pi, ri in zip(batch, p, resp):
            ri = ri.tolist()
            hit = [k for k, t in enumerate(ri) if t in stops]
            ri = ri[: hit[0] + 1] if hit else ri  # keep the first stop token, drop padding after it
            out.append({"id": r["id"], "source": r["source"], "prompt": pi, "response": torch.tensor(ri),
                        "ended": bool(hit)})
        print(f"  {min(i + bs, len(rows))}/{len(rows)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    rc = cfg["rollouts"]
    precise_matmul()
    set_seed(cfg["seed"])
    dev = device()
    tok = load_tokenizer(cfg["model_id"])
    model = load_model(cfg["model_id"], dtype=torch.float32, dev=dev)

    for name in ("new", "old"):
        rows = read_jsonl(out_path(cfg, "data", f"check_{name}.jsonl"))
        print(f"sampling {name}: {len(rows)} prompts")
        res = sample_set(model, tok, rows, rc["batch_size"], rc["max_new_tokens"], dev)
        torch.save(res, out_path(cfg, f"rollouts_{name}.pt"))
        n_tok = sum(len(r["response"]) for r in res)
        print(f"{name}: {n_tok} response tokens, {sum(r['ended'] for r in res)}/{len(res)} ended")
        print("  example:", repr(tok.decode(res[0]["response"])[:300]))


if __name__ == "__main__":
    main()
