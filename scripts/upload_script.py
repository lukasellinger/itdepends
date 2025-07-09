from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "dpo-llama3-all",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True
)

REPO_NAME = "lukasellinger/uncertain-dpo-llama-v3p1-8b-instruct"

# Push model and tokenizer
model.push_to_hub(REPO_NAME)
tokenizer.push_to_hub(REPO_NAME)