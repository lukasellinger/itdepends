import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.lang_map import LANG_MAP
from utils.models import MODELS, MODEL_ORDER, DPO_ORDER

# Use Seaborn style for cleaner plots
sns.set(style="white")


response_types = ['answer_attempt', 'hedge', 'clarification', 'refuse']
colors = {
    'answer_attempt': '#2c7bb6',
    'hedge': '#abd9e9',
    'clarification': '#fdae61',
    'refuse': '#d7191c'
}

def generate_cats_graphs(data, correct_only=False, base_file='cats_all_langs.pdf'):
    languages = list(data.keys())
    settings = set()
    for lang_data in data.values():
        for model_data in lang_data.values():
            settings.update(model_data.keys())
    settings = sorted(settings)

    n_settings = len(settings)
    group_width = 0.7
    bar_width = group_width / n_settings
    spacing = 1
    x = np.arange(len(MODEL_ORDER)) * spacing

    n_langs = len(languages)
    fig_width = 6 * n_langs

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times"],
    })
    fig, axs = plt.subplots(1, n_langs, figsize=(fig_width, 5), sharey=False)

    if n_langs == 1:
        axs = [axs]  # ensure axs is iterable

    for ax, lang in zip(axs, languages):
        for i, setting in enumerate(settings):
            offset = (i - (n_settings - 1) / 2) * bar_width
            bottoms = np.zeros(len(MODEL_ORDER))
            for r_type in response_types:
                heights = [
                    data[lang][model][setting].get(r_type, 0)
                    for model in MODEL_ORDER
                ]
                ax.bar(
                    x + offset,
                    heights,
                    bar_width,
                    bottom=bottoms,
                    label=f'{r_type}' if i == 0 else "",
                    color=colors[r_type]
                )
                bottoms += heights

        # Only English gets y-label
        if lang == "en":
            ax.set_ylabel('Percentage', fontsize=26)
        else:
            ax.set_ylabel('')

        ax.set_title(LANG_MAP.get(lang), fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [MODELS.get(model).get('standard', model) for model in MODEL_ORDER],
            rotation=45,
            ha='right'
        )
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=26)

    plt.tight_layout()
    plt.savefig(f'{base_file}.pdf', bbox_inches='tight')
    plt.close()

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
    group_width = 0.7
    bar_width = group_width / n_settings
    spacing = 1
    x = np.arange(len(languages)) * spacing

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times"],
    })

    fig, axs = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=False)

    if len(models) == 1:
        axs = [axs]  # ensure axs is iterable

    for ax, model in zip(axs, DPO_ORDER):
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
        if model == 'llama-8b':
            ax.set_ylabel('Percentage', fontsize=26)
        ax.set_title(f"{MODELS.get(model).get('dpo')}", fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=26)
    plt.tight_layout()
    plt.savefig(f"{base_file}{'_correct' if correct_only else ''}_dpo.pdf")


def generate_cot_cats_graphs(data, base_file='cats_{lang}'):
    group_width = 0.7
    bar_width = group_width / 2
    spacing = 1
    x = np.arange(2) * spacing

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times"],
    })

    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    settings = ['normal', 'simple']

    for i, setting in enumerate(settings):
        offset = (i - (2 - 1) / 2) * bar_width
        bottoms = np.zeros(2)
        for r_type in response_types:
            keys = ['normal', 'cot_normal'] if setting == "normal" else ['simple', 'cot_simple']

            heights = [
                data[key]['coarse_type']['percentage'].get(r_type, 0)
                for key in keys
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
    ax.set_ylabel('Percentage', fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard', 'CoT'])
    ax.set_ylim(0, 100)
    ax.tick_params(axis='both', which='major', labelsize=26)
    plt.tight_layout()
    plt.savefig(f"{base_file}.pdf")
