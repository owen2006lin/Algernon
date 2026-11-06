from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from huggingface_hub import InferenceClient
from dotenv import load_dotenv


load_dotenv()
client = InferenceClient()


model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)


dataset = load_dataset("openai/gsm8k" , "main")
train = dataset["train"]
test = dataset["test"]

print(train[1]["answer"])



import re
PATTERN = "#### -?(\d+)"

def extract_answer(text):
    match = re.search(PATTERN, text)
    if match: 
        result = match.group(1)
        return result
    else:
        raise ValueError("No match found in the text.")


for example in train:
    try:
        text = example["answer"]
        answer = extract_answer(text)
    except ValueError as e:
        print(f"Error extracting answer from example {example}: {e}")