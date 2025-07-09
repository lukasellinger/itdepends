from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
import torch

# ----------- Config -----------
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUT_DIR = "./dpo-llama3-all"

# ----------- Load tokenizer & model -----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer.pad_token = tokenizer.eos_token
# ----------- Apply LoRA -----------
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

# ----------- DPO Config -----------
dpo_config = DPOConfig(
    beta=0.1,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=2,
    logging_steps=1,
    save_strategy="no",
    report_to=[]
)

def preprocess_dpo(example):
    messages = example["prompt"]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return {
        "prompt": prompt,
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }

dataset = load_dataset('lukasellinger/itdepends-dpo', split="capableof_fly", download_mode="force_redownload")
print(dataset.cache_files)
dataset = dataset.map(preprocess_dpo)

# ----------- Load Trainer -----------
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,  # fallback to base model
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ----------- Train -----------
dpo_trainer.train()

# ----------- Save -----------
dpo_trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

dpo_model = dpo_trainer.model.merge_and_unload()
REPO_NAME = "lukasellinger/uncertain-dpo-llama-v3p1-8b-instruct"

# Push model and tokenizer
dpo_model.push_to_hub(REPO_NAME)
tokenizer.push_to_hub(REPO_NAME)
print("Training complete. Model saved to:", OUTPUT_DIR)
