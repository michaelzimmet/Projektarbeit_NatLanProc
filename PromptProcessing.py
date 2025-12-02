from transformers import AutoTokenizer, AutoModelForCausalLM

def load_prompts():
    with open('prompts_2.txt', 'r') as f:
        prompts = f.readlines()

    return prompts

def write_data_to_file(data: list, file_name:str):
    with open(file_name, "w") as file:
        for i in data:
            file.write(i + "\n")

def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompts = load_prompts()

    answers = []
    count = 1
    print('Start')
    for i in prompts:
        inputs = tokenizer(i, return_tensors="pt")

        outputs = model.generate(inputs['input_ids'],
                                 # max_length=300,
                                 num_return_sequences=1,
                                 pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers.append(response.strip().replace('\n', ''))
        print(count, ' = ', response)
        count += 1

    write_data_to_file(answers, 'answers_2_lama.txt')


if __name__ == '__main__':
    main()

