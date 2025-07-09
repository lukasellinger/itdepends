import itertools
import random

from datasets import Dataset, DatasetDict
from pydantic import BaseModel
from tqdm import tqdm

from config import PROJECT_DIR, Credentials
from data.loader import JSONReader, JSONLineReader
from utils.lang_map import LANG_MAP
from utils.openai_client import prompt_chat_structured

class SatisfiesRelation(BaseModel):
    explanation: str
    satisfies: bool

def does_satisfy_relation(word: str, relation: str) -> bool:
    prompt_content = (
        f"Does the word '{word}' satisfy the relation '{relation}'? "
        "Answer with a brief explanation and either True or False for satisfies."
    )
    return prompt_chat_structured(
        messages=[{'role': 'user', 'content': prompt_content}],
        text_format=SatisfiesRelation,
        temperature=0
    ).satisfies


random.seed(42)

relation_data = {}

relationships = JSONReader().read(f"{PROJECT_DIR}/data/relationships.json")
for relation in relationships:
    contexts_file = f"{PROJECT_DIR}/data/contexts/{relation.get('contexts')}"
    contexts = JSONLineReader().read(contexts_file)
    relation_data[f"{relation.get('questions')}"] = contexts

dataset_1 = []
for question, entities in tqdm(relation_data.items()):
    all_combinations = list(itertools.combinations(entities, 2))

    other_questions = [q for q in relation_data if q != question]

    for combo in tqdm(all_combinations):
        while True:
            other_question = random.choice(other_questions)
            other_entities = relation_data[other_question]
            neg_entity = random.choice(other_entities)
            relation = question.split('questions-')[1].split('.json')[0].replace('-', ' ')
            if does_satisfy_relation(neg_entity['entity'], relation):
                print(f'{neg_entity['entity']} satisfied {relation}. Finding new one.')
                continue
            dataset_1.append({'question': question, 'positive': list(combo), 'negative': neg_entity})
            break

dataset_2 = []
for question, entities in tqdm(relation_data.items()):
    other_questions = [q for q in relation_data if q != question]

    for pos_entity in tqdm(entities):
        while True:
            other_question = random.choice(other_questions)
            other_entities = relation_data[other_question]
            neg_entity = random.choice(other_entities)
            relation = question.split('questions-')[1].split('.json')[0].replace('-', ' ')
            if does_satisfy_relation(neg_entity['entity'], relation):
                print(f'{neg_entity['entity']} satisfied {relation}. Finding new one.')
                continue
            dataset_2.append({'question': question, 'positive': [pos_entity], 'negative': neg_entity})
            break

total_data = {}
for lang in LANG_MAP.keys():
    def extract_lang_version(dataset):
        shared_ref_data = []
        for d in dataset:
            question = JSONReader().read(f'{PROJECT_DIR}/data/questions/{d['question']}').get(lang)
            negative = d['negative']['lang_versions'][lang]
            positive = [p['lang_versions'][lang] for p in d['positive']]
            shared_ref_data.append({'question': question, 'positive': positive, 'negative': negative})
        return Dataset.from_list(shared_ref_data)

    key1 = f'{lang}_shared_ref'
    key2 = f'{lang}_clear_ref'
    total_data[key1] = extract_lang_version(dataset_1)
    total_data[key2] = extract_lang_version(dataset_2)

dataset_dict = DatasetDict(total_data)
dataset_dict.push_to_hub(
    repo_id="lukasellinger/itdepends",
    private=True,
    token=Credentials.hf_api_key
)
