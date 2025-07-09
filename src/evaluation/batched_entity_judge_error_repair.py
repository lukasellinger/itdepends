from datetime import datetime

from config import PROJECT_DIR
from conversation.builder import ConversationBuilder
from data.loader import JSONLineReader
from evaluation.judge import Judge
from utils.openai_client import create_batch_job

SPLITS = ['shared_ref']
MODES = ['normal', 'simple']
LANGS = ['en', 'fr', 'ru', 'zh']
MODELS = ['dpo-llama'] #['gpt-4o-mini', 'gpt-4o', 'deepseek-v3', 'qwen3-32b', 'llama-8b']
ORDERS = [[0, 1, 2], [1, 0, 2], [1, 2, 0], [0, 2, 1], [2, 1, 0], [2, 0, 1]]
JUDGE_FILE = f'{PROJECT_DIR}/data/raw_judge_outputs/entity-batch-dpo.jsonl'

task_ids = [
    "task-shared_ref/en/dpo-llama/outputs-shared_ref-en-dpo-llama-normal-021.jsonl-173",
    "task-shared_ref/fr/dpo-llama/outputs-shared_ref-fr-dpo-llama-normal-012.jsonl-129",
    "task-shared_ref/ru/dpo-llama/outputs-shared_ref-ru-dpo-llama-normal-012.jsonl-113",
    "task-shared_ref/en/dpo-llama/outputs-shared_ref-en-dpo-llama-simple-201.jsonl-47"
]

RESPONSE_FILES = []
for split in SPLITS:
    for mode in MODES:
        for lang in LANGS:
            for model in MODELS:
                for order in ORDERS:
                    if order != [0, 1, 2] and lang != 'en':
                        continue
                    file = f'{split}/{lang}/{model}/outputs-{split}-{lang}-{model}-{mode}-{''.join([str(o) for o in order])}.jsonl'
                    RESPONSE_FILES.append(file)

reader = JSONLineReader()

for response_file in RESPONSE_FILES:
    responses = reader.read(f'{PROJECT_DIR}/data/outputs/' + response_file)
    for idx, response in enumerate(responses):
        custom_id = f"task-{response_file}-{idx}"
        if custom_id not in task_ids:
            continue
        entities = [e['entity'] for e in response.get('entry').get('positive')] + [response.get('entry').get('negative')['entity']]

        judge = Judge("shared_ref", "gpt-4.1-mini-2025-04-14")
        mentioned_entities, explanation = judge.get_mentioned_entities(entities, response['answer'])
        JSONLineReader().write(JUDGE_FILE, [{'custom_id': custom_id, 'response': {'mentioned_entities': mentioned_entities, 'explanation': explanation}}])
