import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.lang_map import LANG_MAP
from utils.models import MODELS, MODEL_ORDER

# Use Seaborn style for cleaner plots
sns.set(style="white")


response_types = ['answer_attempt', 'hedge', 'clarification', 'refuse']
colors = {
    'answer_attempt': '#2c7bb6',
    'hedge': '#abd9e9',
    'clarification': '#fdae61',
    'refuse': '#d7191c'
}

def generate_cats_graphs(data, correct_only=False, base_file='cats_{lang}'):
    languages = list(data.keys())
    settings = set()
    for lang_data in data.values():
        for model_data in lang_data.values():
            settings.update(model_data.keys())
    settings = sorted(settings)

    # Wider x range with spacing between model groups
    n_settings = len(settings)
    group_width = 0.6
    bar_width = group_width / n_settings
    spacing = 0.8
    x = np.arange(len(MODEL_ORDER)) * spacing

    for lang in languages:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times"],
        })
        fig, ax = plt.subplots(figsize=(6, 5))
        # Plot bars
        for i, setting in enumerate(settings):
            offset = (i - (n_settings - 1) / 2) * bar_width
            bottoms = np.zeros(len(MODEL_ORDER))
            for r_type in response_types:
                heights = [
                    data[lang][model][setting].get(r_type, 0)
                    for model in MODEL_ORDER
                ]
                bars = ax.bar(
                    x + offset,
                    heights,
                    bar_width,
                    bottom=bottoms,
                    label=f'{r_type}' if i == 0 else "",
                    color=colors[r_type]
                )
                bottoms += heights

        # Axis formatting
        ax.set_ylabel('Percentage', fontsize=20)
        plt.title(f"{LANG_MAP.get(lang)}", fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels([MODELS.get(model).get('standard', model) for model in MODEL_ORDER])
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"{base_file.format(lang=lang)}{'_correct' if correct_only else ''}.pdf")

def generate_dpo_cats_graphs(data, correct_only=False, base_file='cats_{lang}'):
    languages = list(data.keys())
    settings = set()
    models = set()
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

    for model in models:
        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times"],
        })
        fig, ax = plt.subplots(figsize=(6, 6))

        # Plot bars
        for i, setting in enumerate(settings):
            offset = (i - (n_settings - 1) / 2) * bar_width
            bottoms = np.zeros(len(languages))
            for r_type in response_types:
                heights = [
                    data[lang][model][setting].get(r_type, 0)
                    for lang in languages
                ]
                bars = ax.bar(
                    x + offset,
                    heights,
                    bar_width,
                    bottom=bottoms,
                    label=f'{r_type}' if i == 0 else "",
                    color=colors[r_type]
                )
                bottoms += heights

        # Axis formatting
        ax.set_ylabel('Percentage', fontsize=20)
        plt.title(f"{MODELS.get(model).get('dpo')}", fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.savefig(f"{base_file.format(lang=model)}{'_correct' if correct_only else ''}_dpo.pdf")
