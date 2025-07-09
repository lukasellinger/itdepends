import json
import os
from itertools import permutations
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LANG_STARTERS = {
    'en': 'Provide me one sentence for each of the following: {entity_list}',
    'ru': 'Дайте мне по одному предложению для каждого из следующих слов: {entity_list}',
    'fr': 'Donnez-moi une phrase pour chacun des mots suivants : {entity_list}',
    'ar': 'أعطني جملة واحدة لكل من التالي: {entity_list}',
    'zh': '请为以下每个项目提供一句描述：{entity_list}',
}

ENTITY_JOINER = {
    'en': ', ',
    'ru': ', ',
    'fr': ', ',
    'ar': ' ،',
    'zh': '、 ',
}

MODES = {
    'simple': {
        'en': ' Please answer in simple language.',
        'fr': ' Veuillez répondre dans un langage simple.',
        'ru': ' Пожалуйста, отвечайте простым языком.',
        'ar': ' يرجى الإجابة بلغة بسيطة.',
        'zh': ' 请用通俗易懂的语言回答。',
    },
    'normal': {},
    #'think': ' Think before answering.',
    #'straight': ' Answer straight away.'
}

class DPOConversationBuilder:

    def __init__(self, model, tokenizer, temperature: float = 0.7, mode='normal', order: list[int] = None, lang: str = 'en'):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.mode = mode
        self.order = order or [0, 1, 2]
        self.lang = lang

    def build_conversation(self, entry: dict) -> list[dict]:
        generate_context = self.build_generate_context(entry)
        context = self.build_context(entry)
        mode_addon = MODES.get(self.mode, {}).get(self.lang, "")
        if self.lang == 'ar':
            request = mode_addon + entry.get('question')
        else:
            request = entry.get('question') + mode_addon

        return [
            {"role": "user", "content": generate_context},
            {"role": "assistant", "content": context},
            {"role": "user", "content": request},
        ]

    def build_generate_context(self, entry: dict) -> str:
        positive = entry.get('positive', [])
        negative = entry.get('negative')

        # Get entity names
        entities = [pos['entity'] for pos in positive]
        if negative:
            entities.append(negative['entity'])

        entities = [entities[i] for i in self.order if i < len(entities)]

        entity_list = self.format_entity_list(entities, self.lang)
        return LANG_STARTERS.get(self.lang, '').format(entity_list=entity_list)

    def generate_answer(self, entry: dict) -> dict:
        conversation = self.build_conversation(entry)

        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        answer = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

        return {
            'answer': answer,
            'conversation': conversation,
        }

    @staticmethod
    def format_entity_list(entities: list[str], lang: str = 'en') -> str:
        if len(entities) == 1:
            return entities[0]
        else:
            return ENTITY_JOINER.get(lang, ', ').join(entities)

    def build_context(self, entry: dict) -> str:
        positive = entry.get('positive')
        negative = entry.get('negative')

        cleaned_pos = [pos['context'].rstrip(" .") for pos in positive]
        cleaned_neg = negative['context'].rstrip(" .")

        statements = cleaned_pos + [cleaned_neg]
        statements = [statements[i] for i in self.order if i < len(statements)]

        return ". ".join(statements) + "."


class Reader:
    """General file reader."""

    def __init__(self, encoding="utf-8"):
        self.enc = encoding

    def read(self, file):
        """Read a file."""
        path = Path(file)
        if not path.is_file():
            return None

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

    def __init__(self, pretty_print: bool = False):
        super().__init__()
        self.pretty_print = pretty_print

    def process(self, file):
        """Read each line as json object."""
        data = []
        for line in file.readlines():
            try:
                data.append(json.loads(line.strip()))
            except json.decoder.JSONDecodeError:
                pass

        return data

    def _write(self, file, lines):
        for line in lines:
            if self.pretty_print:
                json.dump(line, file, indent=2, ensure_ascii=False)
            else:
                json.dump(line, file, ensure_ascii=False)
            file.write('\n')

datadict = load_dataset('lukasellinger/itdepends')

model = AutoModelForCausalLM.from_pretrained(
    "lukasellinger/uncertain-dpo-llama-v3p1-8b-instruct",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype="float16",
)
tokenizer = AutoTokenizer.from_pretrained(
    "lukasellinger/uncertain-dpo-llama-v3p1-8b-instruct",#"meta-llama/Meta-Llama-3-8B-Instruct",
    trust_remote_code=True
)

model_name = 'dpo-llama'
split = 'shared_ref'
orders = list(range(2) if split == 'clear_ref' else range(3))

for lang in ['ar']:
    for mode in ['simple', 'normal']:
        dataset = datadict[f'{lang}_{split}']
        for order in [[0, 1, 2]]: #list(permutations(orders)):
            order = list(order)

            output_file = f'data/outputs/{split}/{lang}/{model_name}/outputs-{split}-{lang}-{model_name}-{mode}-{"".join([str(o) for o in order])}.jsonl'
            conv_builder = DPOConversationBuilder(model=model, tokenizer=tokenizer, mode=mode, order=order, lang=lang)

            for entry in tqdm(dataset):
                output = conv_builder.generate_answer(entry)
                print(output)