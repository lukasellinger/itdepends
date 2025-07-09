from datetime import datetime

from config import PROJECT_DIR
from data.loader import JSONLineReader
from evaluation.judge import Judge
from utils.openai_client import create_batch_job

SPLITS = ['clear_ref']
MODES = ['normal', 'simple']
LANGS = ['en', 'fr', 'ar', 'ru', 'zh']
MODELS = ['dpo-llama'] #['gpt-4o-mini', 'gpt-4o', 'deepseek-v3', 'qwen3-32b', 'llama-8b']
ORDERS = [[0, 1], [1, 0]]#, [1, 2, 0], [0, 2, 1], [2, 1, 0], [2, 0, 1]]

RESPONSE_FILES = []
for split in SPLITS:
    for mode in MODES:
        for lang in LANGS:
            for model in MODELS:
                for order in ORDERS:
                    if order != [0, 1] and lang != 'en':
                        continue
                    file = f'{split}/{lang}/{model}/outputs-{split}-{lang}-{model}-{mode}-{''.join([str(o) for o in order])}.jsonl'
                    RESPONSE_FILES.append(file)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
JUDGE_FILE = f'{PROJECT_DIR}/data/judge-inputs/entity-judge-input-{timestamp}.jsonl'
reader = JSONLineReader()

tasks = []
for response_file in RESPONSE_FILES:
    responses = reader.read(f'{PROJECT_DIR}/data/outputs/' + response_file)
    for idx, response in enumerate(responses):
        entities = [e['entity'] for e in response.get('entry').get('positive')] + [response.get('entry').get('negative')['entity']]
        answer = response.get('answer')
        task = {
            "custom_id": f"task-{response_file}-{idx}",
            "method": "POST",
            "url": '/v1/responses',
            "body": {
                "model": "gpt-4.1-mini-2025-04-14",
                "temperature": 0,
                "input": Judge.get_mentioned_entities_instructions(entities, answer),
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "MentionedEntities",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "explanation": {
                                    "type": "string",
                                },
                                "mentioned_entities": {
                                    "type": "array",
                                    "items": {
                                      "type": "string"
                                    },
                                    "description": "List of explicitly mentioned entities from the given list."
                                }
                            },
                            "required": ["explanation", "mentioned_entities"],
                            "additionalProperties": False
                        },
                        "strict": True
                    }
                }
            }
        }
        tasks.append(task)

JSONLineReader().write(JUDGE_FILE, tasks)
create_batch_job(JUDGE_FILE, endpoint='/v1/responses')
