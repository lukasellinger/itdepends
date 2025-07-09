from datasets import load_dataset
from tqdm import tqdm

from config import PROJECT_DIR
from conversation.builder import ConversationBuilder
from data.loader import JSONLineReader
from judge import Judge
from utils.lang_map import LANG_MAP

split = 'clear_ref' #'shared_ref' # 'clear_ref' | 'shared_ref'
mode = 'normal'
order = [1, 0]
model = 'gpt-4o-mini' #'llama4-maverick' #'gpt-4o-2024-08-06' #'gpt-4o-mini-2024-07-18' #'gpt-4o-2024-08-06' #'qwen/qwen-2.5-72b-instruct:free' #'gpt-4o-2024-08-06'
model_path = 'gpt-4o-mini-2024-07-18' #'gpt-4o-2024-08-06' #'gpt-4o-mini-2024-07-18' #'accounts/fireworks/models/llama4-maverick-instruct-basic'
provider = 'openai' #'fireworks'


output_file = f'{PROJECT_DIR}/data/outputs/outputs-{split}-{mode}-{model}-{''.join([str(o) for o in order])}.jsonl'
dataset = load_dataset('lukasellinger/itdepends')
conv_builder = ConversationBuilder(provider=provider, model=model_path, mode=mode, order=order)
judge = Judge(data_type=split)

for lang in LANG_MAP.keys():
    # TODO lang in output file and split
    for entry in tqdm(dataset[split]):
        output = conv_builder.generate_answer(entry)
        JSONLineReader().write(output_file, [{**output, 'entry': entry}])


def print_convo_response(entry: dict):
    for message in entry.get('conversation'):
        print(message.get('content'))
    print(print(entry.get('response')))
