from conversation.builder import ConversationBuilder
from evaluation.judge import Judge

provider = 'openai'
model_path = 'gpt-4o-2024-08-06'
mode = 'normal' # utils/modes.py
order = [0, 1, 2]
lang = 'en' # utils/lang_map.py
data_type = 'shared_ref'


entry = {'positive': [{'entity': 'plane',
                       'context': 'The plane is on the tarmac.'},
                      {'entity': 'bird',
                       'context': 'The bird sits on the tree.'}],
         'negative': {'entity': 'dog',
                      'context': 'The dog wants to go on a walk.'},
         'question': 'Why can it fly?'}

conv_builder = ConversationBuilder(provider=provider, model=model_path, mode=mode, order=order, lang=lang)
judge = Judge(data_type=data_type, )
output = conv_builder.generate_answer(entry)

print(output)
print(judge.judge_response(output))

