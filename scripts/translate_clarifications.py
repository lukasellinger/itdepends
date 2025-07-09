from config import PROJECT_DIR
from data.loader import JSONReader
from utils.lang_map import LANG_MAP
from utils.translate import translate

clarification_questions = JSONReader().read(f'{PROJECT_DIR}/src/evaluation/clarifications.json')
all_translations = []
for entry in clarification_questions:
    lang_entry = dict()
    for lang in LANG_MAP.keys():
        if lang == 'en':
            lang_entry['en'] = {**entry, 'prompt': [{'role': 'user', 'content': entry['prompt']}]}
        else:
            prompt = [{'role': 'user', 'content': translate(entry['prompt'], target_lang=lang.upper())}]
            rejected = translate(entry['rejected'], target_lang=lang.upper())
            chosen = translate(entry['chosen'], target_lang=lang.upper())
            lang_entry[lang] = {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}
    all_translations.append(lang_entry)

JSONReader().write(f'{PROJECT_DIR}/src/evaluation/all_clarifications.json', all_translations)