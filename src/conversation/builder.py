from transformers import pipeline

from config import Credentials
from utils.modes import MODES
from utils.openai_client import prompt_chat

LANG_STARTERS = {
    'en': 'Provide me one sentence for each of the following: {entity_list}',
    'ru': 'Дайте мне по одному предложению для каждого из следующих слов: {entity_list}',
    'fr': 'Donnez-moi une phrase pour chacun des mots suivants : {entity_list}',
    'ar': 'أعطني جملة واحدة لكل من التالي: {entity_list}',
    'zh': '请为以下每个项目提供一句描述：{entity_list}',
}

ENTITY_JOINER = {
    'en': ', ',
    'ru': ', ',
    'fr': ', ',
    'ar': ' ،',
    'zh': '、 ',
}

class ConversationBuilder:

    def __init__(self, provider='openai', model: str = "gpt-4.1-nano-2025-04-14", temperature: float = 0.7, mode='normal', order: list[int] = None, lang: str = 'en'):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.mode = mode
        self.order = order or [0, 1, 2]
        self.lang = lang

    def generate_answer(self, entry: dict) -> dict:
        conversation = self.build_conversation(entry)
        return {
            'answer': prompt_chat(conversation, provider=self.provider, model=self.model, temperature=self.temperature),
            'conversation': conversation,
        }

    def build_conversation(self, entry: dict) -> list[dict]:
        generate_context = self.build_generate_context(entry)
        context = self.build_context(entry)
        mode_addon = MODES.get(self.mode, {}).get(self.lang, "")
        if self.lang == 'ar':
            request = mode_addon + entry.get('question')
        else:
            request = entry.get('question') + mode_addon

        return [
            {"role": "user", "content": generate_context},
            {"role": "assistant", "content": context},
            {"role": "user", "content": request},
        ]

    def build_generate_context(self, entry: dict) -> str:
        positive = entry.get('positive', [])
        negative = entry.get('negative')

        # Get entity names
        entities = [pos['entity'] for pos in positive]
        if negative:
            entities.append(negative['entity'])

        entities = [entities[i] for i in self.order if i < len(entities)]

        entity_list = ConversationBuilder.format_entity_list(entities, self.lang)
        return LANG_STARTERS.get(self.lang, '').format(entity_list=entity_list)

    @staticmethod
    def format_entity_list(entities: list[str], lang: str = 'en') -> str:
        if len(entities) == 1:
            return entities[0]
        else:
            return ENTITY_JOINER.get(lang, ', ').join(entities)

    def build_context(self, entry: dict) -> str:
        positive = entry.get('positive')
        negative = entry.get('negative')

        cleaned_pos = [pos['context'].rstrip(" .") for pos in positive]
        cleaned_neg = negative['context'].rstrip(" .")

        statements = cleaned_pos + [cleaned_neg]
        statements = [statements[i] for i in self.order if i < len(statements)]

        return ". ".join(statements) + "."

class ModelConversationBuilder(ConversationBuilder):

    def __init__(self, model, mode='normal', order: list[int] = None):
        super().__init__(mode=mode, order=order)
        self.pipe = pipeline("text-generation", model=model, token=Credentials.hf_api_key)
        self.mode = mode
        self.order = order or [0, 1, 2]

    def generate_answer(self, entry: dict) -> dict:
        conversation = self.build_conversation(entry)
        answer = self.pipe(conversation, temperature=0.7)[0]['generated_text'][-1]['content']
        return {
            'answer': answer,
            'conversation': conversation,
        }