# ItDepends
This repository contains the code and data for our paper:

[**It Depends: Resolving Referential Ambiguity with Commonsense in Minimal Contexts**](https://aclanthology.org/2025.uncertainlp-main.20/)

> ⚠️ **Note**: The project structure is still a work in progress.

## 🚀 Additional Resources
### 🤖 Fine-tuned Model (DPO)
- Hugging Face: [lukasellinger/uncertain-dpo-llama-v3p1-8b-instruct](https://huggingface.co/lukasellinger/uncertain-dpo-llama-v3p1-8b-instruct)

### 📁 DPO Training Dataset (Preference Pairs)
- Hugging Face: [lukasellinger/itdepends-dpo](https://huggingface.co/datasets/lukasellinger/itdepends-dpo)

### 📊 Evaluation Dataset
- [ItDepends Dataset](https://huggingface.co/datasets/lukasellinger/itdepends)
- 🔍 Evaluation input and results can be found in the `/data/judged_outputs` directory.

## ▶️ Evaluate Your Model
All necessary evaluation scripts can be found in `src/evaluation`.

1. **Set up configuration**  
   Copy the template configuration file and fill in your API keys:
   ```bash
   cp config.py.template config.py
   ```

### 🔹 Single-Sample Evaluation
To test a single example: `run_single_sample.py`

### 🔹 Full Evaluation
We evaluated using the batch api of openai. Therefore, we have multiple steps:
1. Generate Responses (`generate_responses.py`):
   - generate responses for the full dataset on the selected split / language / modes.
   - Splits: defined in `utils/data_type.py`
   - Modes: defined in `utils/modes.py`
   - Languages: defined in `utils/lang_map.py`

2. Judge the Responses:
   - Entity Judge (`batched_entity_judge.py`):
     - Update constants at the top of the script to match your target model, mode, and languages

   - Coarse Response Type Judge (`batched_coarse_judge.py`):
     - Similarly, adjust constants as needed.

    - Parse Judge Outputs (`batched_judge_parse.py`):
      - Set `INPUT_FILE_COARSE` and `INPUT_FILE_ENTITY` to the output files from the previous two steps.

3. Run analysis (`analysis.py`):
    - First, register your model in `utils/models.py`.
    - Then, invoke the desired analysis method.

## Citation

If you use any of the work, please cite the following paper:

```tex
@inproceedings{ellinger-groh-2025-depends,
    title = "It Depends: Resolving Referential Ambiguity in Minimal Contexts with Commonsense Knowledge",
    author = "Ellinger, Lukas  and
      Groh, Georg",
    editor = "Noidea, Noidea",
    booktitle = "Proceedings of the 2nd Workshop on Uncertainty-Aware NLP (UncertaiNLP 2025)",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.uncertainlp-main.20/",
    doi = "10.18653/v1/2025.uncertainlp-main.20",
    pages = "229--246",
    ISBN = "979-8-89176-349-4",
    abstract = "Ambiguous words or underspecified references require interlocutors to resolve them, often by relying on shared context and commonsense knowledge. Therefore, we systematically investigate whether Large Language Models (LLMs) can leverage commonsense to resolve referential ambiguity in multi-turn conversations and analyze their behavior when ambiguity persists. Further, we study how requests for simplified language affect this capacity. Using a novel multilingual evaluation dataset, we test DeepSeek v3, GPT-4o, Qwen3-32B, GPT-4o-mini, and Llama-3.1-8B via LLM-as-Judge and human annotations. Our findings indicate that current LLMs struggle to resolve ambiguity effectively: they tend to commit to a single interpretation or cover all possible references, rather than hedging or seeking clarification. This limitation becomes more pronounced under simplification prompts, which drastically reduce the use of commonsense reasoning and diverse response strategies. Fine-tuning Llama-3.1-8B with Direct Preference Optimization substantially improves ambiguity resolution across all request types. These results underscore the need for advanced fine-tuning to improve LLMs' handling of ambiguity and to ensure robust performance across diverse communication styles."
}
```
