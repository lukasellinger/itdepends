from collections import defaultdict

import pandas as pd
from pylatex import Document, Tabular, NoEscape, Table
from pylatex.utils import bold

from latex.model_translation import MODEL_TRANSLATION


def generate_table(data, table_name):
    doc = Document()
    with doc.create(Table(position='h!')) as table:
        table.append(NoEscape(r'\centering'))
        table.append(NoEscape(r'\setlength{\tabcolsep}{5pt}'))

        with table.create(Tabular('l' + 'r' * 10, width=11, booktabs=True)) as tabular:
            table.append(NoEscape(r'\textbf{Prompt / Model} & \multicolumn{5}{c}{\textbf{Correct}} & \multicolumn{5}{c}{\textbf{Direct}}\\'))
            tabular.add_hline(start=2, end=6, cmidruleoption='lr')
            tabular.add_hline(start=7, end=11, cmidruleoption='lr')

            tabular.add_row(['', 'En', 'Fr', 'Ar', 'Ru', 'Zh',
                             'En', 'Fr', 'Ar', 'Ru', 'Zh',
                             ])
            inverted_data = defaultdict(lambda: defaultdict(dict))
            for lang, lang_data in data.items():
                for model, model_data in lang_data.items():
                    for prompt, prompt_data in model_data.items():
                        inverted_data[prompt][model][lang] = prompt_data

            def highlight_max_and_second(s):
                max_val = s.max()
                result = []
                for v in s:
                    if v == max_val:
                        result.append(bold(f'{v:.2f}'))
                    else:
                        result.append(f'{v:.2f}')
                return result

            for prompt, data in inverted_data.items():
                tabular.add_hline()
                if prompt.lower() == 'simple':
                    tabular.append(NoEscape(r'\multicolumn{11}{l}{\textbf{Prompt: Simple}} \\'))
                elif prompt.lower() == 'normal':
                    tabular.append(NoEscape(r'\multicolumn{11}{l}{\textbf{Prompt: Normal}} \\'))

                # Extract 'value' and 'direct' per model and language to DataFrames for highlighting
                # Build dicts: model -> list of 'value' per lang, model -> list of 'direct' per lang
                models = list(data.keys())
                langs = list(next(iter(data.values())).keys())  # languages from first model

                # Prepare DataFrames: rows=models, cols=langs
                value_matrix = []
                direct_matrix = []
                for model in models:
                    value_row = [data[model][lang]['value'] for lang in langs]
                    direct_row = [data[model][lang]['direct'] for lang in langs]
                    value_matrix.append(value_row)
                    direct_matrix.append(direct_row)

                import pandas as pd
                df_value = pd.DataFrame(value_matrix, index=models, columns=langs)
                df_direct = pd.DataFrame(direct_matrix, index=models, columns=langs)

                # Apply highlighting per column (language)
                highlighted_value_df = df_value.apply(highlight_max_and_second, axis=0)
                highlighted_direct_df = df_direct.apply(highlight_max_and_second, axis=0)

                # Now add rows with highlighted values and directs for each model
                for model in models:
                    row_values = highlighted_value_df.loc[model].tolist()
                    row_directs = highlighted_direct_df.loc[model].tolist()
                    tabular.add_row([MODEL_TRANSLATION.get(model)] + row_values + row_directs)
        table.add_caption('Evaluation scores per prompt type and language')

    doc.generate_tex(table_name)
