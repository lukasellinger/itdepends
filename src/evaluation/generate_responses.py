from itertools import permutations

from datasets import load_dataset
from tqdm import tqdm

from config import PROJECT_DIR
from conversation.builder import ConversationBuilder
from data.loader import JSONLineReader
from utils.lang_map import LANG_MAP
from utils.modes import MODES

model = 'gpt-4o' #'deepseek-v3' #'qwen3-32b' #'llama4-maverick' #'gpt-4o-2024-08-06' #'gpt-4o-mini-2024-07-18' #'gpt-4o-2024-08-06' #'qwen/qwen-2.5-72b-instruct:free' #'gpt-4o-2024-08-06'
model_path = 'gpt-4o-2024-08-06' #'gpt-4o-2024-08-06' #'accounts/fireworks/models/llama-v3p1-8b-instruct' #'qwen/qwen3-32b' #'qwen/qwen3-32b' #'gpt-4o-mini-2024-07-18' #'gpt-4o-2024-08-06' #'gpt-4o-mini-2024-07-18' #'accounts/fireworks/models/llama4-maverick-instruct-basic' accounts/fireworks/models/deepseek-v3
provider = 'openai' #'openrouter' #'openai' #'fireworks'
split = 'shared_ref' #'shared_ref' # choose 'shared_ref', 'clear_ref'
orders = list(range(2) if split == 'clear_ref' else range(3))

datadict = load_dataset('lukasellinger/itdepends')

for lang in ['en']: #LANG_MAP.keys():
    for mode in ['cot_normal', 'cot_simple']:
        dataset = datadict[f'{lang}_{split}']
        for order in [[0, 1, 2]]:#list(permutations(orders)):
            order = list(order)
            #if order != [0, 1]:#, 2]:
            #    continue
            output_file = f'{PROJECT_DIR}/data/outputs/{split}/{lang}/{model}/outputs-{split}-{lang}-{model}-{mode}-{''.join([str(o) for o in order])}.jsonl'
            conv_builder = ConversationBuilder(provider=provider, model=model_path, mode=mode, order=order, lang=lang)

            for entry in tqdm(dataset):
                output = conv_builder.generate_answer(entry)
                JSONLineReader().write(output_file, [{**output, 'entry': entry}])
