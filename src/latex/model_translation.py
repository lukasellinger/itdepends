from pylatex import NoEscape

MODEL_TRANSLATION = {
    'gpt-4o-mini': NoEscape(r"\openai{} 4o-mini"),
    'gpt-4o': NoEscape(r"\openai{} 4o"),
    "qwen3-32b": NoEscape(r"\qwen{} 3-32B"),
    "deepseek-v3": NoEscape(r"\deepseek{} v3"),
    "llama-8b": NoEscape(r"\meta{} 3.1-8B"),
    "dpo-llama": NoEscape("DPO Llama (Ours)"),
}