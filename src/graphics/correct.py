import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.lang_map import LANG_MAP
from utils.models import MODELS

sns.set(style="white")

bar_width = 0.11

def generate_correctness_graphs(data, base_file='correct_predictions_{lang}'):
    languages = list(data.keys())

    models = set()
    settings = set()
    for lang_data in data.values():
        models.update(lang_data.keys())
        for model_data in lang_data.values():
            settings.update(model_data.keys())
    settings = sorted(settings)

    en_models = data.get('en', {})
    models_with_val = [(model, en_models.get(model, {}).get('Normal', {}).get('value')) for model in models]
    models_sorted = [model for model, val in sorted(models_with_val, key=lambda x: x[1], reverse=True)]

    n_models = len(models_sorted)
    x = np.arange(len(settings)) * 0.6

    # Find global max value for y-limit
    all_values = []
    for lang in languages:
        for model in models:
            for setting in settings:
                val = data.get(lang, {}).get(model, {}).get(setting, {}).get('value')
                all_values.append(val)
    max_val = max(all_values) if all_values else 1
    y_limit = max_val * 1.1  # add 10% margin

    for lang in languages:
        plt.figure(figsize=(6, 5))
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times"],
        })
        for j, model in enumerate(models_sorted):
            data_entry = data.get(lang, {}).get(model, {})
            values = [data_entry.get(setting, {}).get('value') for setting in settings]
            offset = (j - n_models / 2) * bar_width + bar_width / 2
            color = MODELS.get(model).get('color')
            bars = plt.bar(x + offset, values, width=bar_width, color=color, label=model)
            direct_values = [data_entry.get(setting, {}).get('direct') for setting in settings]
            for bar, direct_val in zip(bars, direct_values):
                x_bar = bar.get_x() + bar.get_width() / 2
                y_bar = direct_val
                plt.plot([x_bar - bar.get_width() / 3, x_bar + bar.get_width() / 3],
                         [y_bar, y_bar],
                         color='white', linewidth=2, solid_capstyle='round',)

        plt.title(f"{LANG_MAP.get(lang)}", fontsize=26)
        plt.xticks(x, settings, fontsize=20)
        plt.tick_params(axis='x', length=0)
        plt.ylabel("Correct Responses (%)", fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0, y_limit)  # use global y-limit here

        #plt.legend(title="Models", fontsize=10)
        plt.tight_layout()
        plt.savefig(f"{base_file.format(lang=lang)}.pdf")
        plt.close()


def generate_dpo_correctness_graphs(data, base_file='correct_predictions_{lang}'):
    languages = list(data.keys())
    models = set()
    settings = set()
    for lang_data in data.values():
        models.update(lang_data.keys())
        for model_data in lang_data.values():
            settings.update(model_data.keys())
    settings = sorted(settings)

    n_settings = len(settings)
    group_width = 0.6
    bar_width = group_width / n_settings
    spacing = 0.8
    x = np.arange(len(languages)) * spacing

    # Find global max value for y-limit
    all_values = []
    for lang in languages:
        for model in models:
            for setting in settings:
                val = data.get(lang, {}).get(model, {}).get(setting, {}).get('value')
                all_values.append(val)
    max_val = max(all_values) if all_values else 1
    y_limit = max_val * 1.1  # add 10% margin

    for model in models:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times"],
        })
        fig, ax = plt.subplots(figsize=(5, 4))

        for i, setting in enumerate(settings):
            offset = (i - (n_settings - 1) / 2) * bar_width

            heights = [
                data[lang][model][setting].get('value', 0)
                for lang in languages
            ]
            direct_values = [
                data[lang][model][setting].get('direct', 0)
                for lang in languages
            ]
            bars = ax.bar(
                x + offset,
                heights,
                bar_width,
                color=MODELS.get(model).get('color'),
            )

            for bar, direct_val in zip(bars, direct_values):
                x_bar = bar.get_x() + bar.get_width() / 2
                y_bar = direct_val
                ax.plot([x_bar - bar.get_width() / 3, x_bar + bar.get_width() / 3],
                             [y_bar, y_bar],
                            color='white', linewidth=2, solid_capstyle='round',)

        # Axis formatting
        ax.set_ylabel('Correct Responses (%)', fontsize=20)
        plt.title(f"{MODELS.get(model).get('dpo')}", fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.set_ylim(0, y_limit)  # use global y-limit here
        ax.tick_params(axis='both', which='major', labelsize=20)

        plt.tight_layout()
        plt.savefig(f"{base_file.format(lang=model)}-dpo.pdf")
