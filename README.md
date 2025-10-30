# ItDepends
This repository contains the code and data for our paper:

**It Depends: Resolving Referential Ambiguity with Commonsense in Minimal Contexts**

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
@misc{ellinger2025dependsresolvingreferentialambiguity,
      title={It Depends: Resolving Referential Ambiguity in Minimal Contexts with Commonsense Knowledge}, 
      author={Lukas Ellinger and Georg Groh},
      year={2025},
      url={https://arxiv.org/abs/2509.16107},
      annote={Comment: Accepted by UncertaiNLP workshop @ EMNLP 2025} 
}
```
