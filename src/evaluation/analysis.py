from collections import Counter
from itertools import permutations
from typing import Tuple

import numpy as np
import pandas as pd
from spacy.matcher.dependencymatcher import defaultdict

from config import PROJECT_DIR
from data.loader import JSONLineReader, JSONReader
from graphics.cats import generate_cats_graphs, generate_dpo_cats_graphs, \
    generate_cot_cats_graphs  # , #generate_dpo_cats_graphs
from graphics.correct import generate_correctness_graphs, generate_dpo_correctness_graphs, \
    generate_cot_correctness_graphs
from latex.create_table import generate_table
from latex.model_translation import MODEL_TRANSLATION
from utils.lang_map import LANG_MAP
from utils.models import MODELS
from utils.modes import MODES


class Analysis:
    def __init__(self, datatype: str):
        self.datatype = datatype

    def read_data(self, lang: str, base_file: str, model_id: str) -> dict:
        permute = "01" if self.datatype == 'clear_ref' else "012"
        data = dict()
        for mode in MODES.keys():
            model_data = JSONLineReader().read(base_file.format(datatype=self.datatype, lang=lang, model=model_id, mode=mode, permute=permute)) or []
            data[mode] = model_data
        return data

    def get_data(self, model_id: str) -> dict:
        base_file = f'{PROJECT_DIR}/data/judged_outputs/' + '{datatype}/{lang}/{model}/outputs-{datatype}-{lang}-{model}-{mode}-{permute}.jsonl'
        data = dict()
        for lang in LANG_MAP.keys():
            data[lang] = self.read_data(lang, base_file, model_id)
        return data

    def analyze(self, model_id: str) -> Tuple[dict, dict]:
        responses = self.get_data(model_id)
        stats = defaultdict(dict)
        for lang, lang_responses in responses.items():
            for type_, type_responses in lang_responses.items():
                stats[lang][type_] = self.analyze_responses(type_responses)

        summary_stats = defaultdict(lambda: defaultdict(float))
        variances = defaultdict(lambda: defaultdict(list))
        for lang, lang_stats in stats.items():
            for type_, type_responses in lang_stats.items():
                summary_stats['correct'][type_] += type_responses['correct']['percentage'].get('Correct', 0)
                correct_direct = type_responses['fine_category']['count'].get('Direct', 0) / type_responses['correct']['count'].get('Correct', 1)  * 100
                summary_stats['correct_direct'][type_] += correct_direct
                summary_stats['direct'][type_] += type_responses['fine_category']['percentage'].get('Direct', 0)
                variances['correct_direct'][type_].append(correct_direct)

                for category, category_stats in type_responses['coarse_type']['percentage'].items():
                    summary_stats[category][type_] += category_stats

        for k1, v1 in variances.items():
            for k2, v2 in v1.items():
                variances[k1][k2] = np.var(v2, ddof=1)
        for k1, v1 in summary_stats.items():
            for k2, v2 in v1.items():
                summary_stats[k1][k2] = v2 / 5
        summary_stats['variances'] = variances
        return stats, summary_stats

    @staticmethod
    def compute_percentages(sub: dict) -> dict:
        total = sum(sub.values())
        return {
            key: round((value / total) * 100, 2) if total > 0 else 0.0
            for key, value in sub.items()
        }

    def analyze_responses(self, responses: list[dict], permutation: str = "012") -> dict:
        judge_responses = [r['judge_response'] for r in responses]

        correct_counter = Counter(r['correctness'] for r in judge_responses)
        coarse_counter = Counter(r['coarse_type'] for r in judge_responses)
        fine_counter = Counter(r['fine_category'] for r in judge_responses)
        total_counter = Counter((r['coarse_type'], r['fine_category'], r['correctness']) for r in judge_responses)

        correct_coarse_counter = Counter((r['coarse_type']) for r in judge_responses if r['correctness'] == 'Correct')
        direct_coarse_counter = Counter((r['coarse_type']) for r in judge_responses if r['fine_category'] == 'Direct')

        entity_total = defaultdict(int)
        entity_correct = defaultdict(int)
        entity_partial = defaultdict(int)
        entity_wrong = defaultdict(int)

        pos_total = defaultdict(int)
        pos_correct = defaultdict(int)
        pos_partial = defaultdict(int)
        pos_wrong = defaultdict(int)

        for response, judge in zip(responses, judge_responses):
            pos_entities = [e['entity'] for e in response['entry']['positive']]  # [pos_0, pos_1]
            neg_entity = response['entry']['negative']['entity']
            mentioned = judge['mentioned_entities']

            for idx, entity in enumerate(pos_entities + [neg_entity]):
                entity_key = f'entity_{idx}'
                is_mentioned = int(entity in mentioned)

                pos_idx = permutation.index(str(idx))
                pos_key = f'pos_{pos_idx}'

                entity_total[entity_key] += is_mentioned
                pos_total[pos_key] += is_mentioned
                if judge['correctness'] == 'Correct':
                    entity_correct[entity_key] += is_mentioned
                    pos_correct[pos_key] += is_mentioned
                elif judge['correctness'] == 'Partially Correct':
                    entity_partial[entity_key] += is_mentioned
                    pos_partial[pos_key] += is_mentioned
                elif judge['correctness'] == 'Wrong':
                    entity_wrong[entity_key] += is_mentioned
                    pos_wrong[pos_key] += is_mentioned

        return {
            'correct': {
                'count': dict(correct_counter),
                'percentage': self.compute_percentages(correct_counter)
            },
            'coarse_type': {
                'count': dict(coarse_counter),
                'percentage': self.compute_percentages(coarse_counter)
            },
            'fine_category': {
                'count': dict(fine_counter),
                'percentage': self.compute_percentages(fine_counter)
            },
            'correct_coarse': {
                'count': dict(correct_coarse_counter),
                'percentage': self.compute_percentages(correct_coarse_counter)
            },
            'direct_coarse': {
                'count': dict(direct_coarse_counter),
                'percentage': self.compute_percentages(direct_coarse_counter)
            },
            'total':{
                'count': dict(total_counter),
                'percentage': self.compute_percentages(total_counter)
            },
            'entity_counters': {
                'count': {
                    'total': dict(entity_total),
                    'correct': dict(entity_correct),
                    'part': dict(entity_partial),
                    'wrong': dict(entity_wrong),
                },
                'percentage': {
                    'total': self.compute_percentages(entity_total),
                    'correct': self.compute_percentages(entity_correct),
                    'part': self.compute_percentages(entity_partial),
                    'wrong': self.compute_percentages(entity_wrong),
                },
            },
            'pos_counters': {
                'count': {
                    'total': dict(pos_total),
                    'correct': dict(pos_correct),
                    'part': dict(pos_partial),
                    'wrong': dict(pos_wrong),
                },
                'percentage': {
                    'total': self.compute_percentages(pos_total),
                    'correct': self.compute_percentages(pos_correct),
                    'part': self.compute_percentages(pos_partial),
                    'wrong': self.compute_percentages(pos_wrong),
                },
            },
        }

    def ablate_entity_position(self, model_id: str) -> Tuple[dict, dict, dict, dict]:
        data = dict()
        aggregated = {
            'en': {
                'normal': [],
                'simple': []
            }
        }
        for permutation in permutations([0, 1, 2] if self.datatype == 'shared_ref' else [0, 1]):
            permutation_str = ''.join(list([str(n) for n in permutation]))
            base_file = f'{PROJECT_DIR}/data/judged_outputs/' + '{datatype}/{lang}/{model}/outputs-{datatype}-{lang}-{model}-{mode}-' + f'{permutation_str}.jsonl'
            perm_data = self.read_data(lang='en', base_file=base_file, model_id=model_id)
            data[permutation_str] = perm_data
            aggregated['en']['simple'].extend(perm_data['simple'])
            aggregated['en']['normal'].extend(perm_data['normal'])

        total_stats = defaultdict(dict)
        for permutation, modes in data.items():
            for mode, responses in modes.items():
                total_stats[mode][permutation] = self.analyze_responses(responses, permutation)

        for k1, v1 in aggregated.items():
            for k2, v2 in v1.items():
                aggregated[k1][k2] = self.analyze_responses(v2)

        entity_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        pos_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        for mode, permutation in total_stats.items():
            for p, stat in permutation.items():
                entity_counters = stat['entity_counters']['count']
                pos_counters = stat['pos_counters']['count']
                entity_stat = entity_stats[mode]
                pos_stat = pos_stats[mode]
                for name, counts in entity_counters.items():
                    for entity, count in counts.items():
                        entity_stat[name][entity] += count

                for name, counts in pos_counters.items():
                    for pos, count in counts.items():
                        pos_stat[name][pos] += count

        percentage_pos_stats = defaultdict(dict)
        for mode, stats in pos_stats.items():
            for name, stat in stats.items():
                percentage_pos_stats[mode][name] = {
                    'count': stat,
                    'percentage': self.compute_percentages(stat),
                }

        percentage_entity_stats = defaultdict(dict)
        for mode, stats in entity_stats.items():
            for name, stat in stats.items():
                percentage_entity_stats[mode][name] = {
                    'count': stat,
                    'percentage': self.compute_percentages(stat),
                }
        return total_stats, percentage_entity_stats, percentage_pos_stats, aggregated

    def analyze_all(self) -> dict:
        data = dict()
        lang_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        for model in MODELS.keys():
            general_stats, summary_stats = self.analyze(model_id=model)
            data[model] = {'general_stats': general_stats, 'summary_stats': summary_stats}
            for lang, stats in general_stats.items():
                for type_, type_stats in stats.items():
                    lang_stats[lang][type_]['correct'] += type_stats['correct']['percentage'].get('Correct', 0)  * 1/5
                    lang_stats[lang][type_]['direct'] += type_stats['fine_category']['percentage'].get('Direct', 0)  * 1/5
                    lang_stats[lang][type_]['direct_correct'] += type_stats['fine_category']['count'].get('Direct', 0)  / type_stats['correct']['count'].get('Correct', 1)  * 100 * 1/5

        max_direct, min_direct = 0, 100
        for mode, model_stats in data.items():
            for lang, lang_stats in model_stats.get('general_stats').items():
                for type_, type_stats in lang_stats.items():
                    direct = type_stats['fine_category']['percentage'].get('Direct', 0)
                    if max_direct < direct:
                        max_direct = direct
                    if min_direct > direct:
                        min_direct = direct

        return data

    def ablate_entity_position_all(self) -> dict:
        data = dict()
        aggregated = defaultdict(dict)
        for model in MODELS.keys():
            stats, entity_stats, pos_stats, aggregated_stats = self.ablate_entity_position(model_id=model)
            data[model] = {'stats': stats, 'entity_stats': entity_stats, 'pos_stats': pos_stats, 'aggregated_stats': aggregated_stats}
            aggregated[model]['general_stats'] = aggregated_stats
        #simple_df = pd.DataFrame(
        #    {MODEL_TRANSLATION.get(model): vals["pos_stats"]["simple"]['total']['percentage'] for model, vals in data.items()}).T.round(2)
        #normal_df = pd.DataFrame(
        #    {MODEL_TRANSLATION.get(model): vals["pos_stats"]["normal"]['total']['percentage'] for model, vals in data.items()}).T.round(2)
        self.generate_correctness_graph(data=aggregated)
        self.generate_cats_graph(data=aggregated)
        return data

    def check_significance(self, model_id: str):
        pass

    def check_significance_all(self):
        for model in MODELS.keys():
            self.check_significance(model_id=model)

    def generate_cot_graphs(self):
        data = self.analyze_all()['gpt-4o']['general_stats']['en']
        generate_cot_correctness_graphs(data, 'cot_correct_predictions')
        generate_cot_cats_graphs(data, 'cot_cats')

    def generate_correctness_graph(self, dpo: bool = False, data = None, cto: bool = False):
        data = (self.analyze_all() if not dpo else self.analyze_dpo_all()) if data is None else data
        lang_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for model, model_data in data.items():
            if dpo and model not in {'llama-8b', 'dpo-llama'}:
                continue
            elif not dpo and model == 'dpo-llama':
                continue
            if cto and model != 'gpt-4o':
                continue

            stats = model_data['general_stats']
            for lang, lang_stats in stats.items():
                lang_data[lang][model]['Simple']['value'] = lang_stats['simple']['correct']['percentage']['Correct']
                lang_data[lang][model]['Simple']['direct'] = lang_stats['simple']['fine_category']['percentage'].get('Direct', 0)
                lang_data[lang][model]['Normal']['value'] = lang_stats['normal']['correct']['percentage']['Correct']
                lang_data[lang][model]['Normal']['direct'] = lang_stats['normal']['fine_category']['percentage'].get('Direct', 0)
        base_file = self.datatype + '_correct_predictions'

        if dpo:
            generate_dpo_correctness_graphs(lang_data, base_file)
        else:
            generate_correctness_graphs(lang_data, base_file)
            generate_table(lang_data, self.datatype + '_table')

    def generate_cats_graph(self, correct_only: bool = False, dpo: bool =  False, data = None):
        data = (self.analyze_all() if not dpo else self.analyze_dpo_all()) if data is None else data
        lang_data = defaultdict(lambda: defaultdict(dict))
        data_key = 'correct_coarse' if correct_only else 'coarse_type'
        for model, model_data in data.items():
            if dpo and model not in {'llama-8b', 'dpo-llama'}:
                continue
            elif not dpo and model == 'dpo-llama':
                continue

            stats = model_data['general_stats']
            for lang, lang_stats in stats.items():
                lang_data[lang][model]['Simple'] = lang_stats['simple'][data_key]['percentage']
                lang_data[lang][model]['Normal'] = lang_stats['normal'][data_key]['percentage']

        base_file = self.datatype + '_cats'

        if dpo:
            generate_dpo_cats_graphs(lang_data, correct_only, base_file)
        else:
            generate_cats_graphs(lang_data, correct_only, base_file)

    def generate_direct_graph(self):
        data = self.analyze_all()
        lang_data = defaultdict(lambda: defaultdict(dict))
        for model, model_data in data.items():
            stats = model_data['general_stats']
            for lang, lang_stats in stats.items():
                lang_data[lang][model]['Simple'] = lang_stats['simple']['direct_coarse']['percentage']
                lang_data[lang][model]['Normal'] = lang_stats['normal']['direct_coarse']['percentage']
        generate_cats_graphs(lang_data, base_file='direct_cats_{lang}')

    def analyze_dpo(self, model_id: str) -> tuple[dict, dict]:
        TARGET_QUESTION = JSONReader().read(f'{PROJECT_DIR}/data/questions/questions-capableof-fly.json').values()

        responses = self.get_data(model_id)
        filtered_responses = defaultdict(dict)
        for lang, lang_responses in responses.items():
            for type_, type_responses in lang_responses.items():
                filtered_responses[lang][type_] = [tr for tr in type_responses if tr['entry']['question'] not in TARGET_QUESTION]

        stats = defaultdict(dict)
        for mode, langs in filtered_responses.items():
            for lang, lang_responses in langs.items():
                stats[mode][lang] = self.analyze_responses(lang_responses)

        summary_stats = defaultdict(lambda: defaultdict(float))
        for lang, lang_stats in stats.items():
            for type_, type_responses in lang_stats.items():
                summary_stats['correct'][type_] += type_responses['correct']['percentage']['Correct']
                summary_stats['correct_direct'][type_] += type_responses['fine_category']['count']['Direct'] / type_responses['correct']['count']['Correct'] * 100

                for category, category_stats in type_responses['coarse_type']['percentage'].items():
                    summary_stats[category][type_] += category_stats

        for k1, v1 in summary_stats.items():
            for k2, v2 in v1.items():
                summary_stats[k1][k2] = v2 / 5
        return stats, summary_stats

    def analyze_dpo_all(self):
        data = dict()
        for model in MODELS.keys():
            general_stats, summary_stats = self.analyze_dpo(model_id=model)
            data[model] = {'general_stats': general_stats, 'summary_stats': summary_stats}
        return data

if __name__ == '__main__':
    #data = Analysis().analyze_all()
    #print(data)
    ana = Analysis(datatype='shared_ref')
    ana.generate_cot_graphs()
    #print(data)
    #Analysis().check_significance()