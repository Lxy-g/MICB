import json
from pathlib import Path
from matplotlib import pyplot as plt
import tiktoken
import numpy as np
import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer


from captum.attr import (
    FeatureAblation,
    LLMAttribution,
    TextTokenInput,
    LLMAttributionResult,
)

model_name = 'deepseek-chat'
dataset_name = 'frequency'
# open JSON file
with open('dataset/dev_all.json', 'r', encoding='utf-8') as file1, open('dataset/bias_data/bias_frequency_dev.json', 'r', encoding='utf-8') as file2:
    # load JSON data
    dev_data = json.load(file1)
    bias_data = json.load(file2)

# from test_jupyter2 import featureAbalationChat
def get_logprob_chat(input_messages, model_name: str, output_token: str):
   
    client = OpenAI(api_key="Your api key",base_url= "")
    res = []
    max_retries = 5
    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=input_messages,
                max_tokens=1,
                logprobs=True,
                top_logprobs=5
            )
            if response.choices[0].logprobs.content[0].top_logprobs:
                logprob_dict = {}
                for logprobInstance in response.choices[0].logprobs.content[0].top_logprobs:
                    logprob_dict[logprobInstance.token] = logprobInstance.logprob
                if output_token in logprob_dict:
                    print(f"Yes! target token '{output_token}' is in top5: {logprob_dict}")
                    res.append(logprob_dict[output_token])
                else:  # When the target token is not in the top 20 tokens, a value based on experience is taken. This can be optimized through the probability distribution of the vocabulary output by the large language model.
                    print(f"Oh no! target token '{output_token}' is not in top5: {logprob_dict}")
                    res.append(min(logprob_dict.values()) * 2)
                break
            else:
                raise ValueError("logprobs.top_logprobs[0] is None")  # Manually trigger an exception to retry

        except (AttributeError, IndexError, TypeError, ValueError) as e:
            print(f"Error encountered: {e}, retrying... ({retries + 1}/{max_retries})")
            retries += 1
    if retries == max_retries:
        print("Max retries reached, using default value.")
        res.append(-10000)  # If the maximum retries are reached, use a default value

    return np.array(res)


def featureAbalationChat(input_messages,question_index, model_name: str, target=None, base_token=''):
    
    enc = tiktoken.encoding_for_model('gpt-4')
    # extract full_content, question, bias, option
    full_content = input_messages[1]['content']
    question_end_index = full_content.index('Most')
    question = full_content[:question_end_index].strip()
    rest_content = full_content[question_end_index:].strip()
    content_parts = rest_content.split('\n')
    bias = content_parts[0].strip()
    option = content_parts[1].strip()

    #  question, bias, option encode to token id list
    question_token_ids = enc.encode(question)
    bias_token_ids = enc.encode(bias)
    option_token_ids = enc.encode(option)

    #  question, bias, option transfer to token list
    question_tokens = [enc.decode([token_id]) for token_id in question_token_ids]
    bias_tokens = [enc.decode([token_id]) for token_id in bias_token_ids]
    option_tokens = [enc.decode([token_id]) for token_id in option_token_ids]

    input_token_id_list_list = [enc.encode(item['content']) for item in input_messages]
    input_token_list_list = []  # List of lists of tokens
    input_token_list = []  # Flattened list of tokens

    for token_id_list_partial in input_token_id_list_list:
        token_list_partial = [enc.decode([token_id]) for token_id in token_id_list_partial]
        input_token_list_list.append(token_list_partial)
        input_token_list += token_list_partial
    input_token_length = len(input_token_list)

    attr = np.zeros((1, input_token_length))
    ref = get_logprob_chat(input_messages, model_name, target)
    cnt = 0

    segments = [question_tokens, bias_tokens, option_tokens]

    for i, tokens in enumerate(segments):
        modified_messages = [dict(item) for item in input_messages]
        input_token_list_partial = input_token_list_list[1] 

        for j in range(len(tokens)):  
            temp_prompt = ''
            for k in range(len(tokens)):
                if j == k:
                    temp_prompt += base_token  
                else:
                    temp_prompt += tokens[k]

            if i==0:
                modified_messages[1]['content'] = temp_prompt+bias+option
            elif i==1:
                modified_messages[1]['content'] = question+temp_prompt+option
            elif i==2:
                modified_messages[1]['content'] = question+ bias+temp_prompt
            # modified_messages[1]['content'] = temp_prompt
            attr[:, cnt] = ref - get_logprob_chat(modified_messages, model_name, target)
            cnt += 1

    seq_attr = np.sum(attr, axis=0)
    attr_segment = np.zeros((1, 3))  

    start = 0
    for i, tokens in enumerate(segments):
        end = start + len(tokens)
        attr_segment[0, i] = np.average(attr[0, start:end])
        start = end

    seq_attr_segment = np.sum(attr_segment, axis=0)

    return (LLMAttributionResult(torch.tensor(seq_attr), torch.tensor(attr), input_token_list, [target]),
            LLMAttributionResult(torch.tensor(seq_attr_segment), torch.tensor(attr_segment),
                                 ['question_'+question_index, 'bias', 'option'], [target]))




len_json1 = len(dev_data)
len_json2 = len(bias_data)


image = []
save_dir = Path("./image_token/"+model_name+"/"+dataset_name+'/'+"true_to_false")
save_dir.mkdir(parents=True, exist_ok=True)
client = OpenAI(api_key="sk-47bafe20b55e4b17b38e03dd3f16a764",base_url= "https://api.deepseek.com")
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
        max_retries = 5
        retries = 0
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1
                )
                if response.choices[0].message.content:
                    answer = response.choices[0].message.content
                    break
            except (AttributeError, IndexError, TypeError, ValueError) as e:
                print(f"Error encountered: {e}, retrying... ({retries + 1}/{max_retries})")
                retries += 1
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
        retries1 = 0
        max_retries1 = 5
        while retries1 < max_retries1:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    max_tokens=1
                )
                if response.choices[0].message.content:
                    answer_bias = response.choices[0].message.content
                    break
            except (AttributeError, IndexError, TypeError, ValueError) as e:
                print(f"Error encountered: {e}, retrying... ({retries1 + 1}/{max_retries1})")
                retries1 += 1

    if answer == true_answer and answer_bias != true_answer:
        _, attr_res = featureAbalationChat(messages1, i, model_name, target=answer_bias)
        # attr_res.plot_token_attr(show=True)
        fig, _ = attr_res.plot_token_attr(show=False)
        save_path = save_dir / f"image_{i}.png"
        fig.savefig(save_path, bbox_inches="tight")  
        plt.close(fig)
        image.append(fig)
