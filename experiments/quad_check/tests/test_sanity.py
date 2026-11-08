"""CPU tests on a tiny random Qwen2 + LoRA: padding, line scaling, and r(t) -> 1 as t -> 0."""
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM

from quad_check.check import ratios, run_set
from quad_check.common import left_pad, position_ids_from_mask, precise_matmul
from quad_check.lora_scale import set_lora_scale, verify_scaling


def tiny_model():
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    base = Qwen2ForCausalLM(cfg).float().eval()
    lcfg = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "up_proj"],
                      init_lora_weights=False)  # random B, so the update is nonzero
    return get_peft_model(base, lcfg).eval()


def rolls(n=5):
    g = torch.Generator().manual_seed(1)
    return [{"id": str(i), "source": "a" if i % 2 else "b",
             "prompt": torch.randint(1, 97, (3 + i,), generator=g),
             "response": torch.randint(1, 97, (4 + 2 * i,), generator=g)} for i in range(n)]


def test_left_padding_matches_unpadded():
    precise_matmul()
    m = tiny_model()
    seqs = [torch.randint(1, 97, (L,)) for L in (5, 9)]
    ids, attn, offs = left_pad(seqs, 0)
    with torch.no_grad():
        z = m(input_ids=ids, attention_mask=attn, position_ids=position_ids_from_mask(attn)).logits
        for k, s in enumerate(seqs):
            zs = m(input_ids=s[None]).logits[0]
            assert torch.allclose(z[k, offs[k]:], zs, atol=1e-5)


def test_line_scaling():
    m = tiny_model()
    ids, attn, _ = left_pad([torch.randint(1, 97, (7,))], 0)
    v = verify_scaling(m, ids, attn, position_ids_from_mask(attn))
    assert v["t0_vs_off"] == 0.0
    assert v["t1_vs_merged"] < 1e-5
    assert v["t1_vs_t0"] > 1e-3


def test_ratio_tends_to_one():
    precise_matmul()
    m = tiny_model()
    cfg = {"check": {"ts": [1e-3, 1e-2, 1.0], "eps": [1e-3], "tokens_per_batch": 64}}
    rs = rolls()
    res = run_set(m, rs, cfg, 0, "cpu")
    r = ratios(res, cfg["check"]["ts"], 1e-3)
    assert abs(r["0.001"]["r"] - 1) < 0.01
    assert abs(r["0.01"]["r"] - 1) < abs(r["1.0"]["r"] - 1) + 0.05
    assert sum(res["n_tok"]) == sum(len(x["response"]) for x in rs)
    # stable and naive KL agree at t = 1
    assert abs(sum(res["kl"]["1.0"]) - sum(res["kl_direct_tmax"])) < 1e-4 * max(1.0, sum(res["kl"]["1.0"]))
    set_lora_scale(m, 1.0)
