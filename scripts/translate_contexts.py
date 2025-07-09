import re

from config import PROJECT_DIR
from data.loader import JSONLineReader, JSONReader
from utils.lang_map import LANG_MAP
from utils.translate import translate

base_path = f'{PROJECT_DIR}/data/contexts/'

relationships = JSONReader().read(f'{PROJECT_DIR}/data/relationships.json')
for relationship in relationships:
    file = base_path + relationship['contexts']
    contexts = JSONLineReader().read(file)
    new_contexts = []
    for context in contexts:
        new_context = {
            'entity': context['entity'],
            'lang_versions': {
                'en': {'entity': context['entity'], 'context': context['context']}
            }
        }
        for lang in LANG_MAP.keys():
            if lang == 'en':
                continue

            raw = (
                f"<h>{context['entity']}</h>"
                f"<p>{context['context']}</p>"
            )
            translation = translate(raw, source_lang='EN', target_lang=lang.upper())

            def extract_tag(text: str, tag: str) -> str:
                match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
                if not match:
                    raise ValueError(f"Tag <{tag}> not found in translation: '{text}'")
                return match.group(1).strip()

            entity = extract_tag(translation, "h")
            ctx = extract_tag(translation, "p")
            new_context['lang_versions'][lang] = {'entity': entity, 'context': ctx}

        new_contexts.append(new_context)
    JSONLineReader().write(file, new_contexts, mode='w')