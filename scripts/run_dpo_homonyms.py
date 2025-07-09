import json
import os

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

OUTPUT_FILE = "{dataset}-responses-{model}-{prompt_type}.jsonl"
MODEL = "llama-v3p1-8b-instruct"

PROMPT_TEMPLATES = {
    'child_en': "Explain '{word}' like I am 5 years old.",
    'child_ru': "Объясните «{word}» так, будто мне 5 лет.",
    'child_ar': "اشرح ”{word}“ وكأنني في الخامسة من عمري.",
    'child_fr': "Expliquez-moi « {word} » comme si j'avais 5 ans.",
    'child_zh': "请像我只有五岁一样解释「{word}」。",
    'child_w_context_en': "Explain '{word}' like I am 5 years old. Keep in mind that some words have more than one meaning.",
    "child_w_context_ru": "Объясните «{word}» так, как будто мне 5 лет. Пожалуйста, подумайте о различных контекстах или, если нужно, спросите.",
    "child_w_context_ar": "اشرح ”{word}“ كأنني في الخامسة من عمري. يُرجى التفكير في السياقات المختلفة أو طلب أحدها، إذا لزم الأمر.",
    "child_w_context_fr": "Expliquez-moi « {word} » comme si j'avais 5 ans. Pensez aux différents contextes ou demandez-en un, si nécessaire.",
    "child_w_context_zh": "请像我只有五岁一样解释「{word}」。如果这个词有不同的语境，请考虑这些语境，或者询问我需要哪一种。",
    'simple_en': "What is the definition of '{word}' in simple language?",
    'simple_ru': "Каково определение слова «{word}» на простом языке?",
    "simple_ar": "ما هو تعريف ”{word}“ بلغة بسيطة؟",
    "simple_fr": "Quelle est la définition de « {word} » en langage simple ?",
    "simple_zh": "用通俗的语言来说，“{word}”是什么意思？",
    'simple_w_context_en': "What is the definition of '{word}' in simple language? Keep in mind that some words have more than one meaning.",
    "simple_w_context_ru": "Что такое определение «{word}» на простом языке? Пожалуйста, подумайте о различных контекстах или, если нужно, спросите.",
    "simple_w_context_ar": "ما هو تعريف كلمة {word} بلغة بسيطة؟ يُرجى التفكير في السياقات المختلفة أو السؤال عن أحدها، إذا لزم الأمر.",
    "simple_w_context_fr": "Quelle est la définition de « {word} » en langage simple ? Pensez aux différents contextes ou demandez-en un, si nécessaire.",
    "simple_w_context_zh": "用通俗的语言来说，“{word}”是什么意思？如果这个词有不同的语境，请考虑这些语境，或者询问我需要哪一种。",
    'normal_en': "What is the definition of '{word}'?",
    'normal_ru': "Каково определение понятия «{word}»?",
    "normal_ar": "ما هو تعريف ”{word}“؟",
    "normal_fr": "Quelle est la définition du terme « {word} » ?",
    "normal_zh": "「{word}」的定义是什么？",
    'normal_w_context_en': "What is the definition of '{word}'? Keep in mind that some words have more than one meaning.",
    "normal_w_context_ru": "Каково определение «{word}»? Пожалуйста, подумайте о различных контекстах или, если нужно, спросите.",
    "normal_w_context_ar": "ما هو تعريف كلمة {word}؟ يُرجى التفكير في السياقات المختلفة أو السؤال عن أحدها، إذا لزم الأمر.",
    "normal_w_context_fr": "Quelle est la définition du terme « {word} » ? Pensez aux différents contextes ou demandez-en un, si nécessaire.",
    "normal_w_context_zh": "「{word}」的定义是什么？如果这个词有不同的语境，请考虑这些语境，或者询问我需要哪一种。",
}

class Reader:
    """General file reader."""

    def __init__(self, encoding="utf-8"):
        self.enc = encoding

    def read(self, file):
        """Read a file."""
        with open(file, "r", encoding=self.enc) as f:
            return self.process(f)

    def write(self, file, lines, mode='a'):
        """Write lines to file."""
        if os.path.dirname(file):
            os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, mode, encoding=self.enc) as f:
            self._write(f, lines)

    def _write(self, file, lines):
        """Write lines to an opened file."""

    def process(self, file):
        """Process an opened file."""

class JSONLineReader(Reader):
    """Reader for .jsonl files."""

    def process(self, file):
        """Read each line as json object."""
        data = []
        for line in file.readlines():
            data.append(json.loads(line.strip()))

        return data

    def _write(self, file, lines):
        for line in lines:
            json.dump(line, file, ensure_ascii=False)
            file.write('\n')

model = AutoModelForCausalLM.from_pretrained(
    "dpo-llama3-all",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    return_full_text=False,
    do_sample=False,
)

def define_term(prompt: str):
    messages = [
        {"role": "user", "content": prompt}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return pipe(prompt)[0]["generated_text"]

def generate_model_response(homonyms: list[dict], prompt_template: str, output_file: str):
    for entry in homonyms:
        word = entry["word"]
        prompt = prompt_template.format(word=word)
        model_response = define_term(prompt)
        result = {
            "word": word,
            "model_response": model_response,
        }
        JSONLineReader().write(output_file, [result])

def evaluate_dataset(dataset: list[dict], lang: str, prompt_type: str, context: bool, dataset_name: str):
    suffix = f"w_context_{lang}" if context else lang
    prompt_key = f"{prompt_type}_{suffix}"
    prompt = PROMPT_TEMPLATES.get(prompt_key)

    if prompt is None:
        print(f"Prompt not found for key: {prompt_key}")
        return

    output_path = OUTPUT_FILE.format(dataset=dataset_name, model=MODEL, prompt_type=prompt_key)
    generate_model_response(dataset, prompt, output_path)


dataset_name = 'homonym-mcl-wic'
dataset = load_dataset(f'lukasellinger/{dataset_name}')['en'].to_list()

for context in tqdm([False], desc="HoWN Contexts"):
    for prompt_type in tqdm(['simple', 'normal'], leave=False):
        evaluate_dataset(dataset, lang='en', prompt_type=prompt_type, context=context, dataset_name=dataset_name)
