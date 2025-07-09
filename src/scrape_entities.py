import requests

from config import PROJECT_DIR
from data.loader import JSONReader
from utils.spacy_utils import force_noun_lemmatization


def get_entities_capable_of(relation, action, limit=100):
    url = f"https://api.conceptnet.io/query?rel=/r/{relation}&end=/c/en/{action}&limit={limit}"
    response = requests.get(url)
    data = response.json()
    entities = []
    for edge in data.get('edges', []):
        entity = edge['start']['label']
        entities.append(entity)
    entities = [force_noun_lemmatization(entity) for entity in entities]
    return list(set(filter_entities(entities)))

def filter_entities(entities):
    filtered = []
    for entity in entities:
        entity_lower = entity.lower().strip()

        # Skip if more than two words (strict limit)
        if len(entity_lower.split()) > 2:
            continue

        # Single word: no spaces
        if " " not in entity_lower:
            filtered.append(entity.strip())
        # Starts with 'a ' or 'an '
        elif entity_lower.startswith("a "):
            filtered.append(entity[2:].strip())  # Remove 'a ' prefix
        elif entity_lower.startswith("an "):
            filtered.append(entity[3:].strip())  # Remove 'an ' prefix

    return filtered


relations = JSONReader().read(f'{PROJECT_DIR}/data/relationships.json')
for relationship in relations:
    relation = relationship.get('relation')
    action = relationship.get('action')
    entities = get_entities_capable_of(relation, action, limit=1000)
    JSONReader().write(f'{PROJECT_DIR}/data/entities/entities-{relation.lower()}-{action.lower()}.json', entities)
