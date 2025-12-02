import pandas as pd
from datasets import load_dataset
from jupyterlab.semver import valid


def main():
    #download_squad_dataset()
    test_df, validation_df = load_squad_dataset()

    validation_batch = create_random_batch(100, validation_df)
    validation_batch.to_json('squad_validation_batch.json')

    data = parse_data(validation_batch)
    write_data_to_file(data, 'LLM_Results/second_atempt/prompts_2.txt')


def download_squad_dataset():
    splits = {'train': 'plain_text/train-00000-of-00001.parquet',
              'validation': 'plain_text/validation-00000-of-00001.parquet'}

    test_df = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits["train"])
    validation_df = pd.read_parquet("hf://datasets/rajpurkar/squad/" + splits["validation"])

    test_df.to_json('squad_test_data.json')
    validation_df.to_json('squad_validation_data.json')

def load_squad_dataset():
    test_df = pd.read_json('squad_test_data.json')
    val_df = pd.read_json('squad_validation_data.json')
    return test_df, val_df

def create_random_batch(n: int, dataframe: pd.DataFrame):
    return dataframe.sample(n=n, random_state=42)

def parse_data(dataframe: pd.DataFrame):
    prompt_template = ('Can you answer me the following question "%1" based on the following context "%2"? Please structure your answer always in the same format like '
                       'Question ":" Answer". Dont output long Instruction just the answer as short as possible. If its possible only with one word/phrase')

    prompts = []
    for i in dataframe.itertuples():
        question = i.question.strip().replace('\n', ' ')
        context = i.context.strip().replace('\n', ' ')
        prompt = prompt_template.replace('%1', question).replace('%2', context)
        prompts.append(prompt)

    return prompts

def write_data_to_file(data: list, file_name:str):
    with open(file_name, "w") as file:
        for i in data:
            file.write(i + "\n")


if __name__ == '__main__':
    main()