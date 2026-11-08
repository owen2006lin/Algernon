"""Move the weights along the exact line W0 + t * (alpha/r) * B A by scaling LoRA's stored scaling."""
import copy

import torch
from peft.tuners.lora import LoraLayer


def set_lora_scale(model, t, adapter="default"):
    n = 0
    for m in model.modules():
        if isinstance(m, LoraLayer) and adapter in m.scaling:
            if not hasattr(m, "_base_scaling"):
                m._base_scaling = dict(m.scaling)
            m.scaling[adapter] = t * m._base_scaling[adapter]
            n += 1
    if n == 0:
        raise RuntimeError("no LoRA layers found")
    return n


@torch.no_grad()
def verify_scaling(model, ids, attn, pos):
    """Return max |logit| differences: t=0 vs adapter off, t=1 vs merged model."""
    set_lora_scale(model, 0.0)
    z0 = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits.float()
    with model.disable_adapter():
        zoff = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits.float()
    set_lora_scale(model, 1.0)
    z1 = model(input_ids=ids, attention_mask=attn, position_ids=pos).logits.float()
    merged = copy.deepcopy(model).merge_and_unload()
    zm = merged(input_ids=ids, attention_mask=attn, position_ids=pos).logits.float()
    del merged
    m = attn.bool()
    return {
        "t0_vs_off": (z0 - zoff)[m].abs().max().item(),
        "t1_vs_merged": (z1 - zm)[m].abs().max().item(),
        "t1_vs_t0": (z1 - z0)[m].abs().max().item(),  # should be clearly nonzero
    }
