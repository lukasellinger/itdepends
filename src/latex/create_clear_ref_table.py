import pandas as pd
from pylatex import Document, Tabular, NoEscape, Table
from pylatex.utils import bold

# Example data structure (fill this with your real data and optional deltas)
dummy_data = {
    'child': {
        'Llama 3.1 8B': [4.26, 3.05, 0.61, 1.22, 1.22, 35.41],
        'GPT-4o mini': [5.25, 1.23, 1.23, 0.00, 0.00, 35.83],
        'Qwen3-30B A3B': [5.12, 14.63, 6.71, 5.49, 2.44, 42.03],
        'Llama 4 Maverick': [4.04, 1.83, 0.61, 1.22, 0.00, 34.82],
        'DeepSeek v3': [5.51, 1.84, 0.61, 0.61, 0.61, 34.27],
    },
    'simple': {
        'Llama 3.1 8B': [8.32, 28.05, 11.59, 11.59, 4.88, 49.36],
        'GPT-4o mini': [7.95, 23.78, 12.8, 8.54, 2.44, 51.92],
        'Qwen3-30B A3B': [7.25, 51.83, 10.37, 23.17, 18.29, 51.87],
        'Llama 4 Maverick': [8.52, 42.94, 9.82, 22.09, 11.04, 51.61],
        'DeepSeek v3': [8.76, 40.24, 4.27, 20.12, 15.85, 45.13],
    },
    'normal': {
        'Llama 3.1 8B': [10.54, 90.24, 1.83, 52.44, 35.98, 46.90],
        'GPT-4o mini': [10.74, 89.02, 1.22, 46.95, 40.85, 51.20],
        'Qwen3-30B A3B': [8.93, 89.63, 1.22, 46.95, 41.46, 53.47],
        'Llama 4 Maverick': [10.80, 90.85, 0.61, 44.51, 45.73, 45.25],
        'DeepSeek v3': [10.11, 88.41, 0.61, 38.41, 49.39, 44.55],
    }
}


def generate_clear_ref_table(data):
    doc = Document()
    with doc.create(Table(position='ht')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\small'))
        tabular = Tabular(
            'l S[table-format=2.2] S[table-format=2.2] S[table-format=2.2] S[table-format=2.2] S[table-format=2.2] S[table-format=2.2] S[table-format=2.2] S[table-format=2.2]',
            width=9,
            booktabs=True)

        # Add header row
        tabular.add_row([
            bold('Model'),
            bold('FKGL'),
            bold('Sense Aware'),
            bold('Multi. Def.'),
            bold('HeSA'),
            bold('Full'),
            bold('Both'),
            bold('Complete'),
            bold('Covered')
        ])

        for prompt, models in data.items():
            tabular.add_hline()
            if prompt == 'child':
                tabular.append(NoEscape(r'\multicolumn{9}{l}{\textbf{Prompt: ELI5}} \\'))
            elif prompt == 'simple':
                tabular.append(NoEscape(r'\multicolumn{9}{l}{\textbf{Prompt: Simple}} \\'))
            elif prompt == 'normal':
                tabular.append(NoEscape(r'\multicolumn{9}{l}{\textbf{Prompt: Normal}} \\'))

            df = pd.DataFrame(models).T
            def highlight_max(s):
                is_max = s == s.max()
                return [bold(NoEscape(fr'\tablenum{{{v:.2f}}}')) if m else f"{v:.2f}" for v, m in zip(s, is_max)]

            # Apply only to columns 1 to end
            highlighted_df = df.apply(highlight_max, axis=0)

            for model, values in highlighted_df.iterrows():
                tabular.add_row([model] + values.tolist())

        table.append(tabular)
        table.append(NoEscape(r'\label{tab:clear-ref}'))

    doc.generate_tex('clear_ref_overview_table')