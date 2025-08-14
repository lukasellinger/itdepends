import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.colors as mcolors

from utils.lang_map import LANG_MAP
from utils.models import MODELS, MODEL_ORDER, DPO_ORDER

sns.set(style="white")

bar_width = 0.11

def lighten_color(hex_color, amount=0.5):
    """
    Lightens a hex color towards white.
    amount=0 returns original color, amount=1 returns white.
    """
    r, g, b = mcolors.to_rgb(hex_color)  # convert hex to 0–1 RGB
    return (r + (1 - r) * amount,
            g + (1 - g) * amount,
            b + (1 - b) * amount)

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
    y_limit = 100 # add 10% margin

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times"],
    })
    fig, axs = plt.subplots(1, len(languages), figsize=(len(languages) * 6, 5), sharey=False)

    if len(languages) == 1:
        axs = [axs]  # ensure axs is iterable

    for ax, lang in zip(axs, languages):
        for j, model in enumerate(MODEL_ORDER):
            data_entry = data.get(lang, {}).get(model, {})
            total_values = [data_entry.get(setting, {}).get('value', 0) for setting in settings]
            direct_values = [data_entry.get(setting, {}).get('direct', 0) for setting in settings]
            remainder_values = [total - direct for total, direct in zip(total_values, direct_values)]

            offset = (j - n_models / 2) * bar_width + bar_width / 2
            color = MODELS.get(model).get('color')
            light_color = lighten_color(color, amount=0.5)

            # bottom part = direct values
            ax.bar(
                x + offset,
                direct_values,
                width=bar_width,
                color=color,
                label=model if j == 0 else None
            )
            # top part = remainder
            ax.bar(
                x + offset,
                remainder_values,
                width=bar_width,
                bottom=direct_values,
                color=light_color
            )

        ax.set_title(f"{LANG_MAP.get(lang)}", fontsize=26)
        ax.set_xticks(x, settings, fontsize=26)
        if lang == "en":
            ax.set_ylabel("Correct Responses (%)", fontsize=26)
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.set_ylim(0, y_limit)


    #plt.legend(title="Models", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{base_file}.pdf")
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
    y_limit = 100 #* 1.05  # add 10% margin

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times"],
    })
    fig, axs = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=False)

    if len(models) == 1:
        axs = [axs]  # ensure axs is iterable

    for ax, model in zip(axs, DPO_ORDER):
        for i, setting in enumerate(settings):
            offset = (i - (n_settings - 1) / 2) * bar_width

            total_values = [
                data[lang][model][setting].get('value', 0)
                for lang in languages
            ]
            direct_values = [
                data[lang][model][setting].get('direct', 0)
                for lang in languages
            ]
            remainder_values = [
                total - direct for total, direct in zip(total_values, direct_values)
            ]

            base_color = MODELS.get(model).get('color')
            light_color = lighten_color(base_color, amount=0.5)

            # Direct values
            ax.bar(
                x + offset,
                direct_values,
                bar_width,
                color=base_color,
            )

            # Remainder stacked on top
            ax.bar(
                x + offset,
                remainder_values,
                bar_width,
                bottom=direct_values,
                color=light_color,
            )

        # Axis formatting
        if model == "dpo-llama":
            ax.set_ylabel("Correct Responses (%)", fontsize=26)
        ax.set_title(f"{MODELS.get(model).get('dpo')}", fontsize=26)
        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.set_ylim(0, y_limit)  # use global y-limit here
        ax.tick_params(axis='both', which='major', labelsize=26)

    plt.tight_layout()
    plt.savefig(f"{base_file}-dpo.pdf")
    plt.close()


def generate_cot_correctness_graphs(data, base_file='correct_predictions_{lang}'):
    group_width = 0.6
    bar_width = group_width / 2
    spacing = 0.8
    x = np.arange(2) * spacing

    y_limit = 100

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times"],
    })

    plt.figure(figsize=(6, 5))
    ax = plt.gca()

    settings = ['normal', 'simple']

    for i, setting in enumerate(settings):
        offset = (i - (2 - 1) / 2) * bar_width

        keys = ['normal', 'cot_normal'] if setting == "normal" else ['simple', 'cot_simple']

        total_values = [
            data[key].get('correct', {}).get('percentage', {}).get('Correct', 0) for key in keys
        ]
        direct_values = [
            data[key].get('fine_category', {}).get('percentage', {}).get('Direct', 0) for key in keys
        ]
        remainder_values = [
            total - direct for total, direct in zip(total_values, direct_values)
        ]

        base_color = MODELS.get('gpt-4o').get('color')
        light_color = lighten_color(base_color, amount=0.5)

        # Direct values
        ax.bar(
            x + offset,
            direct_values,
            bar_width,
            color=base_color,
        )

        # Remainder stacked on top
        ax.bar(
            x + offset,
            remainder_values,
            bar_width,
            bottom=direct_values,
            color=light_color,
        )


    ax.set_ylabel("Correct Responses (%)", fontsize=26)
    ax.set_xticks(x)
    ax.set_xticklabels(['Standard', 'CoT'])
    ax.set_ylim(0, y_limit)  # use global y-limit here
    ax.tick_params(axis='both', which='major', labelsize=26)

    plt.tight_layout()
    plt.savefig(f"{base_file}.pdf")
    plt.close()
