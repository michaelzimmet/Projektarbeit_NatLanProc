from pathlib import Path

import datasets
from datasets import load_dataset
import pandas as pd

def main():
    ids = load_validation_dataset(Path('squad_validation_batch.csv'))['id'].tolist()
    validation_dataset = load_dataset("rajpurkar/squad")['validation']
    td_b = validation_dataset.select(
        [validation_dataset['id'].index(id_) for id_ in ids if id_ in validation_dataset['id']])

    exact_match_gpt = calc_exact_match_score(Path('LLM_Results/second_atempt/answers_2_gpt.txt'), td_b)
    exact_match_llama = calc_exact_match_score(Path('LLM_Results/second_atempt/answers_2_lama.txt'), td_b)

    f1_gpt = calc_f1_score(Path('LLM_Results/second_atempt/answers_2_gpt.txt'), td_b)
    f1_llama = calc_f1_score(Path('LLM_Results/second_atempt/answers_2_lama.txt'), td_b)

    print('Exact Match GPT: ', exact_match_gpt)
    print('Exact Match LLAMA: ', exact_match_llama)
    print()
    print('F1 GPT: ', f1_gpt)
    print('F1 LLAMA: ', f1_llama)


def calc_exact_match_score(answers_file: Path, td_b):
    td_b_answers = load_answers(answers_file)

    exact_matches = 0
    for i, j in zip(td_b['answers'], td_b_answers):
        answers = i['text']
        answer_model = j.split(':', maxsplit=1)[1].strip()

        if answer_model.lower() in (answer.lower() for answer in answers):
            exact_matches += 1

    exact_match_score = exact_matches / len(td_b)
    return exact_match_score


def calc_f1_score(answers_file: Path, td_b: datasets):
    td_b_answers = load_answers(answers_file)
    f1_scores = []

    for i, j in zip(td_b['answers'], td_b_answers):
        answers = [answer.lower() for answer in i['text']]
        answer_model = j.split(':', maxsplit=1)[1].strip().lower()

        f1 = 0
        for item in answers:
            f1 = max(f1, calc_single_f1_score(answer_model, item))
        f1_scores.append(f1)

    final_f1_score = sum(f1_scores) / len(f1_scores)
    return final_f1_score


def calc_single_f1_score(predicted, answer):
    predicted_tokens = predicted.split()
    tokens = answer.split()

    common_tokens = set(predicted_tokens) & set(tokens)
    tp = len(common_tokens)  # True Positives
    fp = len(predicted_tokens) - tp  # False Positives
    fn = len(tokens) - tp  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def load_validation_dataset(file: Path):
     return pd.read_csv(str(file))


def load_answers(file: Path):
    with file.open('r') as f:
        answers = f.readlines()

    return answers

if __name__ == '__main__':
    main()