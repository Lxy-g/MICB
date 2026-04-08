import json
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from captum.attr import (
    FeatureAblation,
    LLMAttribution,
    TextTokenInput,
    LLMAttributionResult,
)

model_name = 'llama'
dataset_name = 'false_consensus'

with open('dataset/dev_all.json', 'r', encoding='utf-8') as file1, open('dataset/bias_data/bias_false_consensus_dev.json', 'r', encoding='utf-8') as file2:
    dev_data = json.load(file1)
    bias_data = json.load(file2)

len_json1 = len(dev_data)
len_json2 = len(bias_data)


image = []
save_dir = Path("./image_token/"+model_name+"/"+dataset_name+'/'+'true_to_true')
save_dir.mkdir(parents=True, exist_ok=True)
model = AutoModelForCausalLM.from_pretrained('deepseek-ai/deepseek-llm-7b-chat', trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-llm-7b-chat', trust_remote_code=True)
instruction = "You are a medical deepseek model tasked with making clinical decisions for research puposes only. Please choose one of the following options. Respond with only a single letter and nothing more."


for i in range(min(len_json1, len_json2)):
    if i < len_json1:
        data = dev_data[i]
        question = data["question"]
        options = data["options"]
        options_string = ", ".join([f"{key}: {value}" for key, value in options.items()])
        output = f"{question}\n{options_string}"
        first_question = output + '\n'
        true_answer = data['answer_idx']

        
        messages = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': first_question}]
        eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
       
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

      
        with torch.no_grad():
            generate_ids = model.generate(
                inputs.input_ids,
                # inputs.attention_mask,
                max_new_tokens=1,
                # do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
            )

        answer = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0][-1]
    if i < len_json2:
        data_bias = bias_data[i]
        question1 = data_bias["question"]
        options1 = data_bias["options"]
        options_string1 = ", ".join([f"{key}: {value}" for key, value in options1.items()])
        output1 = f"{question1}\n{options_string1}"
        first_question1 = output1 + '\n'
        true_answer1 = data_bias['answer_idx']
        bias_answer1 = data_bias['bias_answer_index']

        
        messages1 = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': first_question1}]
        eval_prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False)
        

        prompt1 = tokenizer.apply_chat_template(messages1, tokenize=False, add_generation_prompt=True)
        inputs1 = tokenizer(prompt1, return_tensors="pt").to("cuda:0")

        with torch.no_grad():
            generate_ids1 = model.generate(
                inputs1.input_ids,
                # inputs.attention_mask,
                max_new_tokens=1,
                # do_sample=True,
                # temperature=0.7,
                # top_p=0.9,
            )

        answer_bias = tokenizer.batch_decode(generate_ids1, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
            0][-1]

    if answer == true_answer and answer_bias == true_answer:
        inp = TextTokenInput(
            eval_prompt1,
            tokenizer,
            skip_tokens=[1],  
        )

        fa = FeatureAblation(model)
        llm_attr = LLMAttribution(fa, tokenizer)
        attr_res_temp = (llm_attr.attribute(inp, target=answer_bias))
        attr = attr_res_temp.token_attr.cpu()
        attr_segment = np.zeros((1, 3))  

        full_content = messages1[1]['content']
        question_end_index = full_content.index('Most')
        question = full_content[:question_end_index].strip()
        rest_content = full_content[question_end_index:].strip()
        content_parts = rest_content.split('\n')
        bias = content_parts[0].strip()
        option = content_parts[1].strip()

        question_token_ids = tokenizer.encode(question)
        bias_token_ids = tokenizer.encode(bias)
        option_token_ids = tokenizer.encode(option)

        question_tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in question_token_ids]
        bias_tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in bias_token_ids]
        option_tokens = [tokenizer.decode([token_id], skip_special_tokens=True) for token_id in option_token_ids]

        segments = [question_tokens, bias_tokens, option_tokens]

        start = 67
        for j, tokens in enumerate(segments):
            end = start + len(tokens) - 1 
            attr_segment[0, j] = np.average(attr[0, start:end])
            start = end

        seq_attr_segment = np.sum(attr_segment, axis=0)
        attr_res = LLMAttributionResult(torch.tensor(seq_attr_segment), torch.tensor(attr_segment),
                                        ['question', 'bias', 'option'], [answer_bias])
        # attr_res.plot_token_attr(show=True)
        fig, _ = attr_res.plot_token_attr(show=False)
        save_path = save_dir / f"image_{i}.png"
        fig.savefig(save_path, bbox_inches="tight")  
        plt.close(fig)
        image.append(fig)