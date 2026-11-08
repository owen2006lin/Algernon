from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import json, os
from tqdm import tqdm

load_dotenv()
client = InferenceClient()



model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map = "cuda", torch_dtype = "auto")

dataset = load_dataset("openai/gsm8k" , "main")
train = dataset["train"]
test = dataset["test"]


instruction = (
    "Solve the following math problem step by step. Keep your reasoning brief. "
    "At the end, write the final numeric answer on its own line in the format: #### <number>"
)

train_messages = [[{"role": "user", "content": f"{instruction}\n\n{item['question']}"}] for item in train]

# Issues with formatting, few shot examples
shots = []
for example in train.select(range(5)):
    shots += [
        {"role": "user", "content" : example["question"]},
        {"role" : "assistant", "content" : example["answer"]}
    ]

test_messages = [shots + [{"role": "user", "content" : f"{instruction}\n\n{item['question']}"}] for item in test]


# Implement batching so inference fits in cache
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(len(test_messages))

batch_size = 16
outputs = []
for i in tqdm(range(0,len(test_messages), batch_size)):
    batch = test_messages[i: i + batch_size]

    # Converts to single string using template specific to the model/tokenizer
    # Add string informing model to start generating
    # Return pt = PyTorch tensors
    # Returns dict of processed outputs instead of raw tensor
    inputs = tokenizer.apply_chat_template(
        batch, add_generation_prompt = True, return_tensors = "pt", return_dict = True, padding = True
    ).to(model.device)

    # Calls model generation function, limits maximum tokens to generate, samples probability instead of greedy decoding,
    # Temperature controls randomness, lower - safe/predictable, higher - creative and wild
    # Nucleus sampling keeps only pool from top 90% of cumulative probability
    out = model.generate(**inputs, max_new_tokens = 1024, do_sample = True, temperature = 0.7, top_p = 0.9)

    # Out is [batch_size, prompt_length + new_tokens] 2D tensor. Select all entries in batch, only new tokens
    new_tokens = out[:, inputs["input_ids"].shape[-1]:]
    outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))


cache_path = "qwen_test_outputs.json"
with open(cache_path, "w") as f:
        json.dump(outputs, f)


import re
PATTERN = r"#### (-?\d+)"

def extract_answer(text):
    match = re.search(PATTERN, text)
    if match: 
        result = match.group(1)
        return result
    else:
        raise ValueError("No match found in the text.")

'''
for example in train:
    try:
        text = example["answer"]
        answer = extract_answer(text)
    except ValueError as e:
        print(f"Error extracting answer from example {example}: {e}")
'''

