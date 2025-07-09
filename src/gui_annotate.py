import click
import questionary
import random
from sklearn.metrics import cohen_kappa_score, accuracy_score

from config import PROJECT_DIR
from src.data.loader import JSONLineReader

VALID_CATEGORIES = ["answer_attempt", "hedge", "clarification", "refuse", "missing"]
VALID_ANSWERS = ["yes", "no"]

def sample_evaluations(evaluations: list[dict], sample_size: int) -> list[dict]:
    if len(evaluations) <= sample_size:
        click.echo(f"Only {len(evaluations)} evaluations available; using all.")
        return evaluations
    random.seed(42)
    return random.sample(evaluations, sample_size)


def save_annotated_evaluations(annotated: list[dict], output_file: str):
    JSONLineReader().write(output_file, annotated)


def calculate_agreement(annotated: list[dict]):
    """Print Cohen’s k and accuracy for each field."""
    auto_coarse = [row["judge_response"]["coarse_type"] for row in annotated]
    human_coarse = [row["human_response"]["coarse_type"] for row in annotated]

    auto_ents = [row["judge_response"]["mentioned_entities"] for row in annotated]
    human_ents = [row["human_response"]["mentioned_entities"] for row in annotated]

    click.echo("\n─ Agreement statistics ──────────────────────────")

    def report(label, auto, human, allowed):
        k = cohen_kappa_score(auto, human, labels=allowed)
        acc = accuracy_score(auto, human)
        click.echo(f"│ {label:<24} k={k:6.3f}   acc={acc:6.3f} │")

    def exact_match_score(label, auto_ents, human_ents):
        """Compute accuracy based on exact set match between predicted and human entity mentions."""
        assert len(auto_ents) == len(human_ents)
        score =  sum(set(pred) == set(gold) for pred, gold in zip(auto_ents, human_ents)) / len(auto_ents)
        click.echo(f"│ {label:<24} Exact Match score={score:6.3f} │")

    report("Coarse type", auto_coarse, human_coarse, VALID_CATEGORIES)
    exact_match_score("Mentioned Entities", auto_ents, human_ents)
    click.echo("──────────────────────────────────────────────────")


@click.command()
@click.option('--input', required=True, help='Input JSONL file with evaluations.')
@click.option('--output', required=True, help='Output JSONL file for annotated evaluations.')
@click.option('--sample-size', default=20, type=int, help='Number of evaluations to sample.')
def annotate(input, output, sample_size):
    reader = JSONLineReader()
    data = reader.read(input)
    sampled = sample_evaluations(data, sample_size)

    annotated = []

    for i, item in enumerate(sampled, 1):
        click.echo(f"\n--- Annotation {i}/{len(sampled)} ---")

        click.echo(f"Pos Words: {', '.join([e['entity'] for e in item['entry']['positive']])}")
        click.echo(f"Neg Words: {item['entry']['negative']['entity']}")
        click.echo(f"\nAssistant's Answer:\n{item['answer']}\n")

        coarse_type = questionary.select(
            "Select coarse answer type:",
            choices=VALID_CATEGORIES
        ).ask()

        entities = [e['entity'] for e in item['entry']['positive']] + [item['entry']['negative']['entity']]
        mentioned_entities = []
        for entity in entities:
            answered = questionary.select(
                f"Was {entity} explicitly mentioned?",
                choices=VALID_ANSWERS
            ).ask()
            if answered == 'yes':
                mentioned_entities.append(entity)

        human_response = {
            "coarse_type": coarse_type,
            "mentioned_entities": mentioned_entities,
        }

        annotated.append({**item, "human_response": human_response})

    save_annotated_evaluations(annotated, output)
    click.echo(f"\nSaved {len(annotated)} human annotations to: {output}")

    calculate_agreement(annotated)



def calculate(files):
    """Calculate correlation from one or more annotated files."""
    reader = JSONLineReader()
    all_evaluations = []

    for file in files:
        evaluations = reader.read(file)
        all_evaluations.extend(evaluations)
        click.echo(f"Loaded {len(evaluations)} annotated evaluations from {file}.")

    click.echo(f"Total evaluations combined: {len(all_evaluations)}")
    calculate_agreement(all_evaluations)

@click.group()
def cli():
    """A simple app for annotating evaluations and calculating correlation."""
    pass

if __name__ == "__main__":
    files = [
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/deepseek-v3/human-annotate-shared_ref-en-deepseek-v3-normal-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/gpt-4o/human-annotate-shared_ref-en-gpt-4o-normal-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/gpt-4o-mini/human-annotate-shared_ref-en-gpt-4o-mini-normal-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/qwen3-32b/human-annotate-shared_ref-en-qwen3-32b-normal-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/llama-8b/human-annotate-shared_ref-en-llama-8b-simple-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/deepseek-v3/human-annotate-shared_ref-en-deepseek-v3-simple-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/gpt-4o/human-annotate-shared_ref-en-gpt-4o-simple-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/gpt-4o-mini/human-annotate-shared_ref-en-gpt-4o-mini-simple-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/qwen3-32b/human-annotate-shared_ref-en-qwen3-32b-simple-012.jsonl",
        f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/llama-8b/human-annotate-shared_ref-en-llama-8b-simple-012.jsonl"
    ]
    calculate(files)
    #annotate()