"""Stage 1: build the SFT data, the check prompt sets and the accuracy-eval sets.

Outputs (under out_dir/data/):
  sft_train.jsonl   GSM8K train, user + assistant messages (off-policy SFT labels)
  check_new.jsonl   GSM8K test prompts (new task)
  check_old.jsonl   ARC-Challenge + MMLU non-math + IFEval prompts (old tasks)
  eval_gsm8k.jsonl  GSM8K test problems with gold numeric answers
  eval_mc.jsonl     ARC-Challenge + MMLU non-math with gold letters
"""
import argparse
import random
import re

from datasets import load_dataset

from .common import load_config, out_path, write_jsonl

GSM8K_INSTRUCTION = (
    "Solve the following math problem step by step. Keep your reasoning brief. "
    "At the end, write the final numeric answer on its own line in the format: #### <number>"
)
MC_REASON_INSTRUCTION = (
    "Answer the following multiple-choice question. Explain your reasoning briefly, "
    "then give the final answer on its own line in the format: Answer: <letter>"
)
MC_LETTER_INSTRUCTION = (
    "Answer the following multiple-choice question. Reply with only the letter of the correct answer."
)
LETTERS = "ABCDEFGH"

# MMLU subjects dropped from the old-task set: math-heavy or quantitative, too close to GSM8K
MMLU_EXCLUDE = {
    "abstract_algebra", "college_mathematics", "elementary_mathematics", "high_school_mathematics",
    "high_school_statistics", "college_physics", "high_school_physics", "conceptual_physics",
    "college_chemistry", "high_school_chemistry", "college_computer_science",
    "high_school_computer_science", "computer_security", "machine_learning",
    "electrical_engineering", "econometrics", "formal_logic", "astronomy",
}


def user(content):
    return [{"role": "user", "content": content}]


def gsm8k_gold(answer):
    return answer.split("####")[-1].strip().replace(",", "")


def gsm8k_target(answer):
    """Reference solution with the <<calc>> annotations stripped; keeps the '#### n' line."""
    body, final = answer.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return f"{body}\n#### {final.strip()}"


def mc_text(question, choices):
    lines = [question.strip(), ""] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(choices)]
    return "\n".join(lines)


def sample(rows, n, rng):
    rows = list(rows)
    if n is None or n >= len(rows):
        return rows
    return rng.sample(rows, n)


def load_arc():
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    out = []
    for i, ex in enumerate(ds):
        labels = list(ex["choices"]["label"])
        if ex["answerKey"] not in labels:
            continue
        out.append({"id": f"arc-{i}", "source": "arc", "question": ex["question"],
                    "choices": list(ex["choices"]["text"]), "gold": LETTERS[labels.index(ex["answerKey"])]})
    return out


def load_mmlu():
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for i, ex in enumerate(ds):
        if ex["subject"] in MMLU_EXCLUDE:
            continue
        out.append({"id": f"mmlu-{i}", "source": "mmlu", "subject": ex["subject"],
                    "question": ex["question"], "choices": list(ex["choices"]),
                    "gold": LETTERS[ex["answer"]]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    d = cfg["data"]
    rng = random.Random(cfg["seed"])

    gsm = load_dataset("openai/gsm8k", "main")
    train, test = list(gsm["train"]), list(gsm["test"])

    sft = [{"messages": user(f"{GSM8K_INSTRUCTION}\n\n{ex['question']}")
            + [{"role": "assistant", "content": gsm8k_target(ex["answer"])}]}
           for ex in sample(train, d["n_sft"], rng)]
    write_jsonl(out_path(cfg, "data", "sft_train.jsonl"), sft)

    test_idx = list(range(len(test)))
    new = [{"id": f"gsm8k-{i}", "source": "gsm8k",
            "messages": user(f"{GSM8K_INSTRUCTION}\n\n{test[i]['question']}")}
           for i in sample(test_idx, d["n_new"], rng)]
    write_jsonl(out_path(cfg, "data", "check_new.jsonl"), new)

    arc, mmlu = load_arc(), load_mmlu()
    ifeval = [{"id": f"ifeval-{ex['key']}", "source": "ifeval", "prompt": ex["prompt"]}
              for ex in load_dataset("google/IFEval", split="train")]

    old = []
    for src, rows in (("arc", arc), ("mmlu", mmlu)):
        for ex in sample(rows, d["n_old"][src], rng):
            old.append({"id": ex["id"], "source": src,
                        "messages": user(f"{MC_REASON_INSTRUCTION}\n\n{mc_text(ex['question'], ex['choices'])}")})
    for ex in sample(ifeval, d["n_old"]["ifeval"], rng):
        old.append({"id": ex["id"], "source": "ifeval", "messages": user(ex["prompt"])})
    write_jsonl(out_path(cfg, "data", "check_old.jsonl"), old)

    ev = [{"id": f"gsm8k-{i}", "messages": user(f"{GSM8K_INSTRUCTION}\n\n{test[i]['question']}"),
           "gold": gsm8k_gold(test[i]["answer"])}
          for i in (test_idx if d["n_eval_gsm8k"] is None else test_idx[: d["n_eval_gsm8k"]])]
    write_jsonl(out_path(cfg, "data", "eval_gsm8k.jsonl"), ev)

    mc = []
    for rows in (arc, mmlu):
        for ex in sample(rows, d["n_eval_mc"], random.Random(cfg["seed"] + 1)):
            mc.append({"id": ex["id"], "source": ex["source"], "n_choices": len(ex["choices"]),
                       "messages": user(f"{MC_LETTER_INSTRUCTION}\n\n{mc_text(ex['question'], ex['choices'])}"),
                       "gold": ex["gold"]})
    write_jsonl(out_path(cfg, "data", "eval_mc.jsonl"), mc)

    print(f"sft={len(sft)} new={len(new)} old={len(old)} eval_gsm8k={len(ev)} eval_mc={len(mc)}")


if __name__ == "__main__":
    main()
