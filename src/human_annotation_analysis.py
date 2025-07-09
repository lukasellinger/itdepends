from config import PROJECT_DIR
from data.loader import JSONLineReader

def main():
    base_file = f"{PROJECT_DIR}/data/judged_outputs/shared_ref/en/qwen3-32b/human-annotate-shared_ref-en-qwen3-32b-normal-012.jsonl"
    data = JSONLineReader().read(base_file)

    for line in data:
        judge = line.get('judge_response')
        human = line.get('human_response')

        if not judge or not human:
            continue

        mismatch = (
            judge.get('coarse_type') != human.get('coarse_type') or
            set(judge.get('mentioned_entities')) != set(human.get('mentioned_entities'))
        )

        if mismatch:
            print(line)

if __name__ == "__main__":
    main()