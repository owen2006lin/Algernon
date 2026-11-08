"""Shared helpers: config, seeding, model/tokenizer loading, padding and position ids."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # QC_OUT lets the cluster point outputs at scratch without editing the config
    cfg["out_dir"] = os.environ.get("QC_OUT", cfg["out_dir"])
    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def out_path(cfg, *parts):
    p = Path(cfg["out_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def adapter_dir(cfg, name):
    """Adapter directory; unlike out_path this does not create anything."""
    return Path(cfg["out_dir"]) / "adapters" / name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def precise_matmul():
    """Strict fp32: TF32 rounding would swamp the eps finite difference."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def native_bf16():
    """True only with hardware bf16 (Ampere+); V100 reports emulated bf16, which is slow."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


def load_tokenizer(model_id):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def stop_token_ids(tok):
    """Qwen instruct ends a turn with <|im_end|>; <|endoftext|> also terminates."""
    ids = {tok.eos_token_id}
    for t in ("<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    return sorted(i for i in ids if i is not None)


def load_model(model_id, dtype=torch.float32, dev=None):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    return model.to(dev or device()).eval()


def position_ids_from_mask(attn):
    """Positions that skip left padding, matching what generate() uses while sampling."""
    return (attn.long().cumsum(-1) - 1).clamp(min=0)


def left_pad(seqs, pad_id):
    """List of 1-D LongTensors -> (ids, attn, offsets) with left padding."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    offsets = []
    for i, s in enumerate(seqs):
        off = L - len(s)
        ids[i, off:] = s
        attn[i, off:] = 1
        offsets.append(off)
    return ids, attn, offsets


def prompt_ids(tok, messages):
    """Chat-templated prompt with the assistant header appended, as token ids."""
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False).input_ids


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
