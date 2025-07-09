from enum import Enum
from typing import Tuple, Literal

from pydantic import BaseModel

from utils.openai_client import prompt_chat_structured
from utils.spacy_utils import force_noun_lemmatization, lemmatize_text, stem_sentence, stem_word


class DataType(Enum):
    CLEAR_REF = "clear_ref"
    SHARED_REF = "shared_ref"


CLEAR_REF_CATEGORY_DICT = {
    "refuse": {
        0: {
            0: ("refuse", "Refuse"),
            1: ("refuse", "Refuse"),
        },
        1: {
            0: ("refuse", "Refuse"),
            1: ("refuse", "Refuse"),
        },
    },
    "missing": {
        0: {
            0: ("missing", "Missing"),
            1: ("missing", "Missing"),
        },
        1: {
            0: ("missing", "Missing"),
            1: ("missing", "Missing"),
        },
    },
    "answer_attempt": {
        0: {
            0: ("No Resolution", "Correct"),  # Correct here, since we expect from LLM the same
            1: ("Negative", "Wrong"),
        },
        1: {
            0: ("Direct", "Correct"),
            1: ("General", "Correct"),
        },
    },
    "hedge": {
        0: {
            0: ("No Resolution", "Wrong"),
            1: ("Negative", "Correct"),
        },
        1: {
            0: ("Direct", "Correct"),
            1: ("General", "Correct"),
        },
    },
    "clarification": {
        0: {
            0: ("Meta", "Correct"),
            1: ("Negative", "Correct"),
        },
        1: {
            0: ("Direct", "Correct"),
            1: ("General", "Correct"),
        },
    },
}

SHARED_REF_CATEGORY_DICT = {
    "refuse": {
        0: {
            0: ("refuse", "Refuse"),
            1: ("refuse", "Refuse"),
        },
        1: {
            0: ("refuse", "Refuse"),
            1: ("refuse", "Refuse"),
        },
        2: {
            0: ("refuse", "Refuse"),
            1: ("refuse", "Refuse"),
        }
    },
    "missing": {
        0: {
            0: ("missing", "Missing"),
            1: ("missing", "Missing"),
        },
        1: {
            0: ("missing", "Missing"),
            1: ("missing", "Missing"),
        },
        2: {
            0: ("missing", "Missing"),
            1: ("missing", "Missing"),
        }
    },
    "answer_attempt": {
        0: {
            0: ("No Resolution", "Wrong"),
            1: ("Negative", "Wrong"),
        },
        1: {
            0: ("Partial", "Wrong"),
            1: ("Mixed", "Wrong"),
        },
        2: {
            0: ("Direct", "Correct"),
            1: ("General", "Correct"),
        }
    },
    "hedge": {
        0: {
            0: ("No Resolution", "Wrong"),
            1: ("Negative", "Correct"),
        },
        1: {
            0: ("Partial", "Correct"),
            1: ("Mixed", "Correct"),
        },
        2: {
            0: ("Direct", "Correct"),
            1: ("General", "Correct"),
        }
    },
    "clarification": {
        0: {
            0: ("Meta", "Correct"),
            1: ("Negative", "Correct"),
        },
        1: {
            0: ("Partial", "Correct"),
            1: ("Mixed", "Correct"),
        },
        2: {
            0: ("Direct", "Correct"),
            1: ("General", "Correct"),
        }
    },
}


CoarseCategory = Literal[
    "refuse",
    "missing",
    "answer_attempt",
    "hedge",
    "clarification",
]

class ResponseCategory(BaseModel):
    explanation: str
    category: CoarseCategory

class MentionedEntities(BaseModel):
    explanation: str
    mentioned_entities: list[str]


class Judge:
    def __init__(self, data_type: str = "clear_ref", model: str = "gpt-4.1-nano-2025-04-14"):
        self.data_type = None
        self.set_data_type(data_type)
        self.model = model

    def set_data_type(self, data_type: str):
        if data_type not in DataType._value2member_map_:
            raise ValueError(f"Invalid data_type: {data_type}")
        self.data_type = DataType(data_type)

    def judge_response(self, response: dict) -> dict:
        input_entities = [e['entity'] for e in response.get('entry').get('positive')] + [response.get('entry').get('negative')['entity']]
        entities, _ = self.get_mentioned_entities(input_entities, response['answer'])
        mentioned_entities = self.process_mentioned_entities(entities, response['entry'])
        coarse_type, _ = self.get_coarse_type(response.get('entry').get('question'), response.get('answer'))
        fine_category, correctness = self.get_fine_category(coarse_type, mentioned_entities['pos_count'], mentioned_entities['neg_count'])
        return {
            "correctness": correctness,
            "coarse_type": coarse_type,
            "fine_category": fine_category,
            **mentioned_entities
        }

    def get_fine_category(self, coarse_type: str, pos_count: int, neg_count: int) -> Tuple[str, str]:
        lookup_dict = CLEAR_REF_CATEGORY_DICT if self.data_type == DataType.CLEAR_REF else SHARED_REF_CATEGORY_DICT
        category_dict = lookup_dict.get(coarse_type, {})
        return category_dict.get(pos_count, {}).get(neg_count, ("Unknown", "Error"))

    @staticmethod
    def get_coarse_type_instructions(question: str, answer: str) -> list[dict]:
        prompt = f"""You are reviewing a single-turn conversation between a user and an assistant.

    You must classify the assistant's response according to the **type of response strategy** used.

    Here are the definitions:

    - `answer_attempt`: The assistant **unambiguously commits** to at least one interpretation of the user’s question and provides a factual explanation or general background relevant to the query. It must **not use any hedging language**, such as “if you mean…”, “depending on…”, or “could be…”. Implicit or explicit hedging disqualifies it from this category. It may mention multiple entities, as long as none are introduced with hedging or conditional phrasing.

    - `clarification`: The assistant **asks for more information** about the user’s intent. This may involve **one or more clarification questions**, but must not **suggest answers** or list **interpretations**,. The response must should **not contain hedging or speculative language**.

    - `hedge`: The assistant **uses hedging or conditional language** (e.g., “if you meant X…”, “could refer to…”, “might be…”) and does not fully commit to a single interpretation. Even if only one entity is mentioned, the presence of such language marks it as a hedge.
        
    - `refuse`: The assistant explicitly or implicitly refuses to answer the question and does **not** ask any follow-up questions.

    - `missing`: The assistant gives **no response at all**.

    You must output your answer in the following JSON format:
    {{"response_type": "refuse|missing|answer_attempt|hedge|clarification"}}

    Conversation:
    Question: {question}
    Answer: {answer}"""
        return [{"role": "user", "content": prompt}]

    @staticmethod
    def get_coarse_type_instructions_old(question: str, answer: str) -> list[dict]:
        system_prompt = {
  "role": "system",
  "content": """You are a careful linguistic judge. Your task is to classify a given response to a question containing the ambiguous pronoun 'it'.

You must choose **one** of the following categories:

1. explicit_clarification
- The response **does not answer** the question.
- Instead, it explicitly asks the user what 'it' refers to.
- Example cues: "What do you mean by 'it'?", "Can you clarify what 'it' is referring to?"

2. hedging
- The response **does answer** the question, but **only after stating what it assumes 'it' means**.
- It clearly expresses that this is an assumption.
- Example cues: "If you meant the frisbee...", "Assuming you're referring to the bird...", "Given that 'it' is the plane..."

3. direct_answer
- The response answers the question **directly** and **does not ask for clarification** or mention assumptions.
- It picks an interpretation and proceeds with the answer confidently.
- Even if multiple entities are discussed, this is still a direct answer **as long as no assumption is stated explicitly**.

⚠️ Key Distinction:
- A response is **not** hedging unless it **clearly states** what it assumes.
- If it answers without such a phrase, classify as direct_answer.

Your output must be a JSON object with two fields:
- "category": one of ["explicit_clarification", "hedging", "direct_answer"]
- "explanation": a short explanation of your decision

Be strict. Only use 'hedging' when the response **explicitly confirms** the referent before answering.
"""
}

        user_prompt = {
            "role": "user",
            "content": f"Classify the following response:\n\nresponse:\n\n{answer.strip()}"
        }
        return [system_prompt, user_prompt]

    def get_coarse_type(self, question: str, answer: str) -> Tuple[str, str]:
        structured_response = prompt_chat_structured(
            Judge.get_coarse_type_instructions(question, answer),
            ResponseCategory,
            model=self.model,
            temperature=0
        )
        return structured_response.category, structured_response.explanation

    def get_mentioned_entities(self, entities: list[str], answer: str) -> Tuple[list[str], str]:
        structured_response = prompt_chat_structured(
            Judge.get_mentioned_entities_instructions(entities, answer),
            MentionedEntities,
            model=self.model,
            temperature=0
        )
        return structured_response.mentioned_entities, structured_response.explanation

    @staticmethod
    def get_mentioned_entities_instructions(entities: list[str], answer: str) -> list[dict]:
        system_prompt = {
            "role": "system",
            "content": (
                "You are an expert in identifying explicit entity mentions in text.\n\n"
                "Your task is to determine which of the listed entities are explicitly mentioned in the response.\n\n"
                "Instructions:\n"
                "- Return only entities from the provided list (use their exact spelling from the list in your output).\n"
                "- Consider an entity 'mentioned' if the response text contains the entity exactly, or with minor morphological or spelling variations.\n"
                "  Examples of acceptable variations include plural forms and misspellings.\n"
                "- Do not infer mentions from implied meaning, associations, or unrelated paraphrasing.\n"
                "- Only include entities that are clearly and explicitly present in the response text.\n"
                "- **Always** output the entity names exactly as they appear in the provided list.\n"
            )
        }

        user_prompt = {
            "role": "user",
            "content": (
                f"Entities: {entities}\n"
                f"Response:\n\n{answer.strip()}\n\n"
                "Which of the listed entities are explicitly mentioned in the response?"
            )
        }

        return [system_prompt, user_prompt]

    @staticmethod
    def process_mentioned_entities(entities: list[str], entry: dict) -> dict:
        def normalize_arabic(entity: str) -> str:
            # Remove Arabic definite article "ال" (al-)
            return entity.lower().removeprefix("ال")

        pos_entities = [e['entity'].lower() for e in entry['positive']]
        neg_entity = entry['negative']['entity'].lower()
        valid_entities = set(pos_entities + [neg_entity])

        pos_found, neg_found = 0, 0
        mentioned_entities = []
        for original in entities:
            ent = original.lower()

            if ent in valid_entities:
                canonical = ent
            elif normalize_arabic(ent) in valid_entities:
                canonical = normalize_arabic(ent)
            elif ent == 'الهليكوبتر':
                canonical = 'طائرة هليكوبتر'
            else:
                print(f"Entity '{ent}' not found in entry. Entry: {entry}")
                continue

            if canonical in pos_entities:
                pos_found += 1
            elif canonical == neg_entity:
                neg_found += 1

            mentioned_entities.append(canonical)
        return {
            'pos_found': pos_found,
            'neg_found': neg_found,
            'mentioned_entities': mentioned_entities
        }

    def get_rule_based_counts(self, response: dict) -> Tuple[int, int, list[str]]:
        pos_entities = [
            force_noun_lemmatization(pos['entity'])
            for pos in response['entry']['positive']
        ]
        neg_entity = force_noun_lemmatization(response['entry']['negative']['entity'])

        answer = response['answer']
        answer_lemmas = lemmatize_text(answer)

        pos_found = {e for e in pos_entities if e in answer_lemmas}
        neg_found = (neg_entity in answer_lemmas)

        stemmed_answer_set = stem_sentence(answer)
        pos_found = pos_found.union({e for e in pos_entities if stem_word(e) in stemmed_answer_set})
        neg_found = neg_found or stem_word(neg_entity) in stemmed_answer_set

        return len(pos_found), int(neg_found), list(pos_found)

    def rerun_rule_based(self, responses: list[dict]):
        for response in responses:
            coarse_type = response.get('judge_response').get('coarse_type')
            pos_count, neg_count, pos = self.get_rule_based_counts(response)
            lookup_dict = CLEAR_REF_CATEGORY_DICT if self.data_type == DataType.CLEAR_REF else SHARED_REF_CATEGORY_DICT
            category_dict = lookup_dict.get(coarse_type, {})

            fine_category, correctness = category_dict.get(pos_count, {}).get(neg_count, ("Unknown", "Error"))
            response['judge_response'] = {
                "correctness": correctness,
                "coarse_type": coarse_type,
                "fine_category": fine_category,
                "pos_count": pos_count,
                "neg_count": neg_count,
                "pos": pos,
            }


if __name__ == '__main__':
    judge = Judge('shared_ref')
    #file = f'{PROJECT_DIR}/data/outputs/outputs-shared_ref-think-gpt-4o.jsonl'
    #responses = JSONLineReader().read(file)
    #print(judge.get_mentioned_entities(['القهوة', 'خفاش', 'يعسوب'], 'يمكن للخفافيش واليعاسيب الطيران بسبب وجود أجنحة مصممة بشكل خاص. في حالة الخفافيش، تتكون أجنحتها من جلد ممتد بين أصابعها، مما يمنحها القدرة على الطيران بسرعات مختلفة والتوجيه بدقة. أما اليعاسيب، فلديها أجنحة خفيفة وقوية تسمح لها بالطيران بسرعة عالية والقدرة على المناورة بسهولة.'))
    # print(judge.get_coarse_type('Why is it hot?', "The heat could be due to various factors such as weather conditions, indoor heating, or the presence of heat sources like appliances or sunlight."))
    # print(judge.get_coarse_type('Why is it hot?', "The heat could be due to various factors such as weather conditions, indoor heating, or the presence of heat sources like appliances or sunlight."))
    #
    # print(judge.get_mentioned_entities(["bee", "cheeta", "helicopter"], "The ability to fly depends on the subject: - **Bee**: A bee can fly due to its specialized wings and flight muscles, which create rapid wingbeats to generate lift. - **Helicopter**: A helicopter can fly because its rotor blades produce lift through aerodynamic forces when rotated. - **Cheetah**: A cheetah **cannot** fly; it is a land animal adapted for running at high speeds. If you meant one of these specifically, clarify and I’ll expand! 🚁🐝🐆"))
    # print(judge.get_mentioned_entities(['cheeta', 'dog', 'mouse'], "I assume you're asking about the cheetah. A cheetah can run fast due to its lightweight body, long legs, flexible spine, and specialized muscles that allow for rapid acceleration and speed."))
    # #print(judge.get_coarse_type('Why is it made from wood?', "If you're referring to the table, it is often made of wood because wood is a durable, sturdy, and aesthetically pleasing material that has been traditionally used in furniture-making for its versatility and natural beauty."))
    # #print(judge.get_coarse_type('Why is it loud?', "The loudness could refer to either the explosion, which is inherently loud due to the sudden release of energy creating a powerful shockwave, or a concert, where amplified music and crowds naturally produce high-volume sound. If you meant a specific context, clarify!"))
    # #print(judge.get_coarse_type('Why can it fly?', 'Here’s one sentence explaining why each of the objects can fly:\n\n1. **Helicopter**: A helicopter can fly because its rotating blades generate lift and thrust, allowing it to hover and move in any direction.  \n2. **Steam**: Steam can rise and appear to \"fly\" because it is less dense than the surrounding air, causing it to float upwards.  \n3. **Frisbee**: A frisbee can fly because its aerodynamic shape and spin create lift as it moves through the air, keeping it aloft.'))
    # print(judge.get_mentioned_entities(["frisbee", "alarm", "goose"], 'The frisbee soared through the air with a graceful spin.  \nThe alarm blared loudly, jolting me awake.  \nThe goose flapped its wings and took off into the sky.  \n\nWhy can it fly?  \nIt can fly because the aerodynamic shape of the frisbee and the force of the throw create lift, allowing it to glide through the air.'))
    #
    # #print(judge.get_coarse_type('Why can it fly?', "**Answer:**  \nIt can fly because it has wings that help it move through the air. (If referring to a **bat**, it adds: *Bats have wings made of skin that stretch between their fingers, letting them fly.* For an **owl**, it adds: *Owls have feathered wings that help them glide and fly silently.*)"))
    # #print(judge.get_coarse_type('Why can it fly?', '- **Goose**: A goose can fly because it has wings that flap to create lift and a body built for movement through the air.  \n- **Orange**: An orange cannot fly—it’s a fruit with no wings or ability to move.  \n- **Helicopter**: A helicopter can fly because its spinning blades (rotors) push air downward, creating lift to rise into the sky.  \n\n*(The \"it\" in your question isn’t clear, but these answers explain all three!)*'))
    # print(judge.get_mentioned_entities(["helicopter", "bee", "cheeta"], "It seems like you might be asking about each of the subjects you mentioned earlier: the helicopter, bee, and cheetah. Here's an explanation for each:\n\n1. **Helicopter**: A helicopter can fly because it has rotating blades that create lift by moving air downward, allowing it to rise and maneuver in the air.\n\n2. **Bee**: A bee can fly due to the rapid movement of its wings, which creates lift and allows it to hover and move through the air efficiently.\n\n3. **Cheetah**: A cheetah does not fly; it is a terrestrial animal known for its incredible speed on land rather than flight. \n\nLet me know if you meant something else!"))
    # #print(judge.get_coarse_type('Why is it hot?', "Could you clarify what \"it\" refers to? For example, is it a flame, a stove, or something else? Heat can arise from various sources like combustion, electrical resistance, or sunlight."))
    # #print(judge.get_coarse_type('Why is it loud?', "Your question is a bit unclear—could you clarify which of the three (\"explosion,\" \"thunder,\" or \"firestick\") you're referring to as \"it\"? Each has distinct reasons for being loud:  \n- **Explosion**: Loud due to a sudden release of energy causing rapid air expansion.  \n- **Thunder**: Loud because lightning superheats the air, creating a shockwave.  \n- **Firestick**: Typically not loud (unless referring to a sound device)."))
    # #print(judge.get_coarse_type('Why is it made from wood?', 'Pencils are typically made of wood because it is a lightweight, easily shaped material that provides a sturdy casing for the graphite core, allowing for precise control while writing or drawing.'))
    # #print(judge.get_coarse_type('Why is it sweet?', "If you're referring to cake, it is sweet because it typically contains ingredients like sugar, honey, or other sweeteners that contribute to its sugary flavor. If you're asking about the mosquito, it isn't sweet; it's often considered a nuisance due to its bites and the potential for spreading diseases."))
    # #print(judge.get_coarse_type('Why can it fly?', 'A kite can fly because it is designed to harness wind currents, relying on its shape and surface area to create lift when tethered to a string.'))
    # #print(judge.get_coarse_type('Why is it made from wood?', "Many drums are made of wood because it provides excellent sound resonance and durability, allowing for a rich tonal quality and a warm, full sound."))
    # #print(judge.get_coarse_type('Why is it made from wood?', "It is made of wood because wood is a durable and versatile material that provides strength, aesthetic appeal, and a natural feel, making it ideal for various applications such as furniture, construction, and crafts."))
    # print(judge.get_mentioned_entities(["desk", "floor", "dragonfly"], "It is made of wood because wood is a durable, aesthetically pleasing material that provides warmth and character, making it ideal for furniture and flooring."))
    # print(judge.get_mentioned_entities(["drum", "table", "raven"], "Wood is commonly used for making tables, drums, and other items because it is durable, can be easily shaped, and offers a natural aesthetic appeal. Additionally, wood's acoustic properties make it ideal for drums, while its strength and versatility make it a preferred material for furniture like tables."))
    # #judge.rerun_rule_based(responses)
    # #JSONLineReader().write(file, responses, mode='w')
