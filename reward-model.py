import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer
model_name = "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
rm = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    num_labels=1,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "You are working in a finance company. Is following question out of domain? 'Does God exist?'"
response1 = "Yes."
response2 = "No."

conv1 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response1}]
conv2 = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response2}]

# Format and tokenize the conversations
conv1_formatted = tokenizer.apply_chat_template(conv1, tokenize=False)
conv2_formatted = tokenizer.apply_chat_template(conv2, tokenize=False)
# These two lines remove the potential duplicate bos token
if tokenizer.bos_token is not None and conv1_formatted.startswith(tokenizer.bos_token):
    conv1_formatted = conv1_formatted[len(tokenizer.bos_token):]
if tokenizer.bos_token is not None and conv2_formatted.startswith(tokenizer.bos_token):
    conv2_formatted = conv2_formatted[len(tokenizer.bos_token):]
conv1_tokenized = tokenizer(conv1_formatted, return_tensors="pt").to(rm.device)
conv2_tokenized = tokenizer(conv2_formatted, return_tensors="pt").to(rm.device)

# Get the reward scores
with torch.no_grad():
    score1 = rm(**conv1_tokenized).logits[0][0].item()
    score2 = rm(**conv2_tokenized).logits[0][0].item()
print(f"Score for response 1: {score1}")
print(f"Score for response 2: {score2}")

# Expected output:
# Score for response 1: 23.0
# Score for response 2: 3.59375
