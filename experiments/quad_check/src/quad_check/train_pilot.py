"""Stage 2: LoRA SFT of Qwen2.5-0.5B-Instruct on GSM8K (off-policy reference answers).

Loss is on assistant tokens only. Saves the adapter at each fraction in train.save_fracs
to out_dir/adapters/step_<pct> (and step_100 is also copied to adapters/final).
"""
import argparse
import csv
import math
import random
import shutil
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from .common import adapter_dir, device, load_config, load_model, load_tokenizer, native_bf16, out_path, read_jsonl, set_seed


def encode(tok, messages, max_len):
    """Token ids and labels; prompt tokens (system+user+assistant header) are masked with -100."""
    full = tok.apply_chat_template(messages, tokenize=False)
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
    assert full.startswith(prompt), "chat template is not prefix-consistent"
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    r_ids = tok(full[len(prompt):], add_special_tokens=False).input_ids
    ids = p_ids + r_ids
    if len(ids) > max_len:
        return None
    return ids, [-100] * len(p_ids) + r_ids


def collate(batch, pad_id):
    L = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (x, y) in enumerate(batch):  # right padding for training
        ids[i, : len(x)] = torch.tensor(x)
        labels[i, : len(y)] = torch.tensor(y)
        attn[i, : len(x)] = 1
    return ids, labels, attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    tc = cfg["train"]
    set_seed(cfg["seed"])
    dev = device()

    tok = load_tokenizer(cfg["model_id"])
    rows = read_jsonl(out_path(cfg, "data", "sft_train.jsonl"))
    data = [e for e in (encode(tok, r["messages"], tc["max_len"]) for r in rows) if e is not None]
    print(f"{len(data)}/{len(rows)} examples within max_len={tc['max_len']}")

    bf16 = native_bf16()
    model = load_model(cfg["model_id"], dtype=torch.bfloat16 if bf16 else torch.float32, dev=dev)
    lcfg = LoraConfig(r=tc["lora_r"], lora_alpha=tc["lora_alpha"], lora_dropout=tc["lora_dropout"],
                      target_modules=tc["target_modules"], use_rslora=False, bias="none",
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)  # adapters are kept in fp32 (autocast_adapter_dtype)
    model.print_trainable_parameters()
    model.train()

    mb, acc = tc["micro_batch"], tc["grad_accum"]
    steps_per_epoch = math.ceil(len(data) / (mb * acc))
    total = steps_per_epoch * tc["epochs"]
    save_at = {max(1, round(f * total)): f"step_{round(f * 100)}" for f in tc["save_fracs"]}

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, max(1, round(tc["warmup_frac"] * total)), total)

    log_path = out_path(cfg, "train_log.csv")
    log = open(log_path, "w", newline="")
    w = csv.writer(log)
    w.writerow(["step", "loss", "lr", "tokens", "elapsed_s"])

    rng = random.Random(cfg["seed"])
    step, tokens, t0 = 0, 0, time.time()
    for epoch in range(tc["epochs"]):
        order = list(range(len(data)))
        rng.shuffle(order)
        chunks = [order[i : i + mb * acc] for i in range(0, len(order), mb * acc)]
        for chunk in chunks:
            n_tok = sum(sum(1 for y in data[i][1] if y != -100) for i in chunk)
            loss_sum = 0.0
            for j in range(0, len(chunk), mb):
                ids, labels, attn = (x.to(dev) for x in collate([data[i] for i in chunk[j : j + mb]], tok.pad_token_id))
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                    logits = model(input_ids=ids, attention_mask=attn).logits
                # token-level mean over the whole optimizer step
                loss = torch.nn.functional.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1),
                    ignore_index=-100, reduction="sum") / n_tok
                loss.backward()
                loss_sum += loss.item()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            tokens += n_tok
            w.writerow([step, f"{loss_sum:.4f}", f"{sched.get_last_lr()[0]:.3e}", tokens, f"{time.time() - t0:.1f}"])
            if step % 10 == 0 or step == 1:
                log.flush()
                print(f"step {step}/{total} loss {loss_sum:.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{time.time() - t0:.0f}s", flush=True)
            if step in save_at:
                d = adapter_dir(cfg, save_at[step])
                model.save_pretrained(d)
                print(f"saved {d}")
    log.close()
    final = adapter_dir(cfg, "final")
    shutil.rmtree(final, ignore_errors=True)
    shutil.copytree(adapter_dir(cfg, "step_100"), final)
    print(f"done: {step} steps, {tokens} target tokens, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
