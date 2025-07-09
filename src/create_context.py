from tqdm import tqdm

from config import PROJECT_DIR
from context.generator import ContextGenerator
from data.loader import JSONReader, JSONLineReader

generator = ContextGenerator()

relationships = JSONReader().read(f"{PROJECT_DIR}/data/relationships.json")

for relation in tqdm(relationships, desc="Processing relationships"):
    entity_file = f"{PROJECT_DIR}/data/entities/{relation.get('entities')}"
    entities = JSONReader().read(entity_file)

    context_path = f"{PROJECT_DIR}/data/contexts/{relation.get('contexts')}"
    output_lines = []

    for entity in tqdm(entities, desc=f"Generating contexts for '{relation.get('entities')}'", leave=False):
        context = generator.generate_context(entity, relation.get('action'))
        output_lines.append({'entity': entity, 'context': context})

    JSONLineReader().write(context_path, output_lines)