import json

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(json_path: str = 'dataset/dev_all.json'):
    with open(json_path, 'r') as file:
        question_all = json.load(file)
    return question_all


if __name__ == '__main__':

    question_all = load_dataset()
    correct = 0
    total = len(question_all)
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    instruction = "You are a medical llama model tasked with making clinical decisions for research purposes only. Please choose one of the following options. Respond with only a single letter and nothing more."

    if "o1-mini" in model_name or "Deepseek" in model_name:
        client = OpenAI(api_key="YOUR_API_KEY",
                        base_url="YOUR_BASE_URL", )
    if "deepseek" in model_name or "Llama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    for data in question_all:

        question = data["question"]
        options = data["options"]
        options_string = ", ".join([f"{key}: {value}" for key, value in options.items()])
        output = f"{question}\n{options_string}"
        first_question = output + '\n'
        true_answer = data['answer_idx']
        messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': first_question}]

        if "o1-mini" in model_name or "Deepseek" in model_name:
            max_retries = 5
            retries = 0
            while retries < max_retries:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=messages,
                        max_tokens=1
                    )

                    target = response.choices[0].message.content
                    if target:
                        break
                except (AttributeError, IndexError, TypeError, ValueError) as e:
                    print(f"Error encountered: {e}, retrying... ({retries + 1}/{max_retries})")
                    retries += 1
            if retries == max_retries:
                print("Max retries reached, using default value.")
                target = "A"



        if "deepseek" in model_name or "Llama" in model_name:
            model.eval()
            # eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                outputs_ori = model(**inputs, output_attentions=True)
            logits = outputs_ori["logits"]
            probs = torch.softmax(logits[:, -1], dim=1)
            logits_orig, answer_t = torch.max(probs, dim=1)
            target = tokenizer.decode(answer_t, skip_special_tokens=True)

        if target == true_answer:
            correct = correct + 1
        else:
            continue

    accuracy = correct / total
    print(accuracy)