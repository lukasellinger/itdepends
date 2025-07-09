import random
from collections import defaultdict
from itertools import permutations

from datasets import Dataset, DatasetDict

from config import PROJECT_DIR
from conversation.builder import ConversationBuilder
from data.loader import JSONLineReader, JSONReader
from utils.lang_map import LANG_MAP
from utils.models import MODELS
from utils.modes import MODES
from utils.translate import translate

random.seed(42)

def shuffle_entry(entry: dict, datatype = 'shared_ref'):
    permuts = list(permutations([0, 1, 2] if datatype == 'shared_ref' else [0, 1]))
    random_permut = random.choice(permuts)
    builder = ConversationBuilder(order=random_permut)
    conversation = builder.build_conversation(entry['entry'])
    entry['conversation'] = conversation
    return entry

def load_shared_ref_data(mode: str, target_questions: list[str]):
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model in MODELS.keys():
        for perm in list(permutations(['0', '1', '2'])):
            file = f'{PROJECT_DIR}/data/judged_outputs/shared_ref/en/{model}/outputs-shared_ref-en-{model}-{mode}-{''.join(perm)}.jsonl'
            results = reader.read(file)
            for idx, result in enumerate(results):
                if result['entry']['question'] not in target_questions:
                    continue
                if model == TARGET_MODEL:
                    all_results[idx]['en']['source'].append(result)
                all_results[idx]['en']['candidates'].append(result)

        for lang in LANG_MAP.keys():
            if lang == 'en':
                continue
            file = f'{PROJECT_DIR}/data/judged_outputs/shared_ref/{lang}/{model}/outputs-shared_ref-{lang}-{model}-{mode}-012.jsonl'
            results = reader.read(file)
            for idx, result in enumerate(results):
                if result['entry']['question'] not in target_questions:
                    continue
                if model == TARGET_MODEL:
                    all_results[idx][lang]['source'].append(result)
                else:
                    all_results[idx][lang]['candidates'].append(result)
    return all_results

def load_clear_ref_data(mode: str, target_questions: list[str]):
    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for model in MODELS.keys():
        for perm in list(permutations(['0', '1'])):
            file = f'{PROJECT_DIR}/data/judged_outputs/clear_ref/en/{model}/outputs-clear_ref-en-{model}-{mode}-{''.join(perm)}.jsonl'
            results = reader.read(file)
            for idx, result in enumerate(results):
                if result['entry']['question'] not in target_questions:
                    continue
                if model == TARGET_MODEL:
                    all_results[idx]['en']['source'].append(result)
                all_results[idx]['en']['candidates'].append(result)

        for lang in LANG_MAP.keys():
            if lang == 'en':
                continue
            file = f'{PROJECT_DIR}/data/judged_outputs/clear_ref/{lang}/{model}/outputs-clear_ref-{lang}-{model}-{mode}-01.jsonl'
            results = reader.read(file)
            for idx, result in enumerate(results):
                if result['entry']['question'] not in target_questions:
                    continue
                if model == TARGET_MODEL:
                    all_results[idx][lang]['source'].append(result)
                else:
                    all_results[idx][lang]['candidates'].append(result)
    return all_results

reader = JSONLineReader()
TARGET_MODEL = 'llama-8b'
TARGET_QUESTION = JSONReader().read(f'{PROJECT_DIR}/data/questions/questions-capableof-fly.json').values()
DATA_TYPE = 'shared-ref'

dpos = []
train_words = defaultdict(set)
coarse_counter = defaultdict(lambda: defaultdict(int))
for mode in MODES.keys():
    model_outputs = load_shared_ref_data(mode, TARGET_QUESTION)
    entry_counter = 0
    for output in model_outputs.values():
        counted = False
        correct_candidates = {lang: [candidate for candidate in lang_output['candidates'] if
                                     (candidate['judge_response']['fine_category'] == 'Direct' or candidate['judge_response']['coarse_type'] == 'clarification')]  for lang, lang_output in output.items()}
        wrong_sources = {lang: [source for source in lang_output['source'] if
                         source['judge_response']['correctness'] == 'Wrong'] for lang, lang_output in output.items()}

        for lang, wrong_lang_output in wrong_sources.items():
            if len(wrong_lang_output) != 0 and len(correct_candidates[lang]) > 0:
                continue
            if len(correct_candidates['en']) > 0:
                clarifications = [c for c in correct_candidates['en'] if c['judge_response']['coarse_type'] == 'clarification']
                clarification = None
                if clarifications:
                    clarification = clarifications[0]
                    correct_candidates[lang].append({**clarification,
                                                     'answer': 'asdf'}) #translate(clarification['answer'], source_lang='EN', target_lang=lang.upper())})  # translate('asdf', source_lang='EN', target_lang=lang.upper())})

                alternatives = [c for c in correct_candidates['en']if c != clarification ]
                if len(alternatives) != 0:
                    chosen_candidate = random.choice(alternatives)
                    correct_candidates[lang].append({**chosen_candidate, 'answer': 'asdf'}) # translate(chosen_candidate['answer'], source_lang='EN', target_lang=lang.upper())}) #translate('asdf', source_lang='EN', target_lang=lang.upper())})

        for lang in LANG_MAP.keys():
            lang_output = output[lang]
            wrong_lang_sources = wrong_sources[lang]
            correct_lang_candidates = correct_candidates[lang]

            if wrong_lang_sources and correct_lang_candidates:
                wrong_source = wrong_lang_sources[0] # one wrong source per lang is enough, I guess
                if not counted:
                    entry_counter += 1
                    counted = True
            else:
                continue

            #wrong_source['conversation'][-1]['content'] = wrong_source['conversation'][-1]['content'] + ' /no_think'
            sampled_correct_candidates = random.sample(correct_lang_candidates, min(2, len(correct_lang_candidates)))
            for correct_candidate in sampled_correct_candidates:
                wrong_source = shuffle_entry(wrong_source, 'shared_ref')
                coarse_counter[lang][correct_candidate['judge_response']['coarse_type']] += 1
                dpo = {
                    "entry": wrong_source['entry'],
                    "prompt": wrong_source['conversation'],
                    "chosen": correct_candidate["answer"],
                    "rejected": wrong_source['answer']
                    #"chosen": {'role': 'assistant', 'content': '<think>\n\n</think>\n\n' + correct_candidate["answer"]},
                    #"rejected": {'role': 'assistant', 'content': '<think>\n\n</think>\n\n' + wrong_source["answer"]},
                }
                dpos.append(dpo)

    model_outputs = load_clear_ref_data(mode, TARGET_QUESTION)
    clear_ref_entry_counter = 0
    for output in model_outputs.values():
        counted = False
        correct_candidates = {lang: [candidate for candidate in lang_output['candidates'] if
                                     candidate['judge_response']['fine_category'] == 'Direct'] for
                              lang, lang_output in output.items()}
        wrong_sources = {lang: [source for source in lang_output['source'] if
                                source['judge_response']['fine_category'] not in ['Direct', "No Resolution"]] for lang, lang_output in
                        output.items()}

        for lang in LANG_MAP.keys():
            lang_output = output[lang]
            wrong_lang_sources = wrong_sources[lang]
            correct_lang_candidates = correct_candidates[lang]

            if wrong_lang_sources and correct_lang_candidates:
                wrong_source = wrong_lang_sources[0]  # one wrong source per lang is enough, I guess
                if not counted:
                    clear_ref_entry_counter += 1
                    counted = True
            else:
                continue

            sampled_correct_candidates = random.sample(correct_lang_candidates, min(2, len(correct_lang_candidates)))
            for correct_candidate in sampled_correct_candidates:
                wrong_source = shuffle_entry(wrong_source, 'shared_ref')
                coarse_counter[lang][correct_candidate['judge_response']['coarse_type']] += 1
                dpo = {
                    "entry": wrong_source['entry'],
                    "prompt": wrong_source['conversation'],
                    "chosen": correct_candidate["answer"],
                    "rejected": wrong_source['answer']
                    # "chosen": {'role': 'assistant', 'content': '<think>\n\n</think>\n\n' + correct_candidate["answer"]},
                    # "rejected": {'role': 'assistant', 'content': '<think>\n\n</think>\n\n' + wrong_source["answer"]},
                }
                dpos.append(dpo)

    print(len(dpos))
    print('SharedRef:', entry_counter)
    print('ClearRef:', clear_ref_entry_counter)

clarification_questions = JSONReader().read(f'{PROJECT_DIR}/src/evaluation/all_clarifications.json')
for lang in LANG_MAP.keys():
    dpos.extend([{'entry': {'type': 'extra_clarification'}, **c[lang]} for c in clarification_questions])

print(coarse_counter)
dataset = Dataset.from_list(dpos)
dataset_dict = DatasetDict({'capableof_fly': dataset})
#dataset_dict.push_to_hub(
#    repo_id="lukasellinger/itdepends-dpo",
#)