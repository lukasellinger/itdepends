from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

from utils.lang_map import LANG_MAP
from utils.openai_client import prompt_chat_structured

class ContextModel(BaseModel):
    sentence: str

class ContextGenerator:
    def __init__(self, lang: str = 'en', model: str = "gpt-4.1-nano-2025-04-14", temperature: float = 0.1):
        self.lang = lang
        self.model = model
        self.temperature = temperature

    @retry(stop=stop_after_attempt(10))
    def generate_context(self, word: str, action: str):
        prompt = self._build_prompt(word, action)
        messages = [{"role": "user", "content": prompt}]

        sentence = prompt_chat_structured(messages, ContextModel, model=self.model,
                                          temperature=self.temperature).sentence

        if action in sentence:
            if action in word:
                if sentence.count(action) > 1:
                    print(f"Warning: Generated sentence contains the attribute '{action}' multiple times: {sentence}")
                    raise Exception("Attribute appeared too often, retrying...")
            else:
                print(f"Warning: Generated sentence contains the attribute '{action}': {sentence}")
                raise Exception("Attribute found in sentence, retrying...")

        return sentence

    def _build_prompt(self, word: str, action: str) -> str:
        lang_part = f" Answer in {LANG_MAP[self.lang]}." if self.lang != 'en' else ''
        instruction = (
            f"Generate one short, neutral sentence that starts with 'A/An {word}'. "
            f"Use '{word}' as a noun in the sense related to the property '{action}', "
            f"but do not describe, refer to, or imply the property '{action}' in any way. "
            f"Avoid verbs, phrases, or settings that are typically associated with '{action}'. "
            f"Instead, write a simple, unrelated sentence that still uses the correct meaning of the word."
            f"{lang_part}"
        )
        return instruction

if __name__ == "__main__":
    context = ContextGenerator('ar').generate_context("owl", "fly")
    print(context)