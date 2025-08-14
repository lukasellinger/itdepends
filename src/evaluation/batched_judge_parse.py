import json
import re
from collections import defaultdict

from tqdm import tqdm

from config import PROJECT_DIR
from data.loader import JSONLineReader
from evaluation.judge import Judge

judge = Judge(data_type='shared_ref')
INPUT_FILE_COARSE = f"{PROJECT_DIR}/data/raw_judge_outputs/coarse-batch-cot.jsonl"
INPUT_FILE_ENTITY = f"{PROJECT_DIR}/data/raw_judge_outputs/entity-batch-cot.jsonl"
reader = JSONLineReader()
coarse_judge_results = reader.read(INPUT_FILE_COARSE)
entity_judge_results = reader.read(INPUT_FILE_ENTITY)

entity_results_by_id = {
    item["custom_id"]: item
    for item in entity_judge_results
}

responses = defaultdict(list)
for coarse_judge_result in tqdm(coarse_judge_results):
    custom_id = coarse_judge_result["custom_id"]
    match = re.match(r"task-(.*)-(\d+)", custom_id)
    if match:
        response_file = match.group(1)
        idx = int(match.group(2))
    else:
        raise Exception(f"Invalid judge result: {coarse_judge_result}")

    try:
        coarse_message = coarse_judge_result['response']['body']['output'][-1]['content'][-1]['text']
        coarse_type = json.loads(coarse_message)['category']
    except Exception:
        print(f'Could not parse response for id {idx} {response_file} - {coarse_message}')
        continue

    entity_result = entity_results_by_id[custom_id]
    try:
        response = entity_result['response']
        if 'body' in response:
            entity_message = entity_result['response']['body']['output'][-1]['content'][-1]['text']
            entities = json.loads(entity_message)['mentioned_entities']
        else:
            entities = response['mentioned_entities']
    except Exception:
        print(f'Could not parse response for id {idx} {response_file} - {entity_message}')
        continue

    response = reader.read(f'{PROJECT_DIR}/data/outputs/' + response_file)[idx]

    processed_entities = judge.process_mentioned_entities(entities, response['entry'])
    fine_category, correctness = judge.get_fine_category(coarse_type, processed_entities["pos_found"], processed_entities["neg_found"])
    response = {"judge_response": {
        "correctness": correctness,
        "coarse_type": coarse_type,
        "fine_category": fine_category,
        **processed_entities,
    }, **response}
    responses[response_file].append(response)
for response_file, responses in responses.items():
    reader.write(f'{PROJECT_DIR}/data/judged_outputs/' + response_file, responses)
