import numpy as np
import torch
import tiktoken
from openai import OpenAI
import json
from PIL import Image, ImageDraw, ImageFont
import colorsys
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline

from captum.attr import (
    FeatureAblation,
    LLMAttribution,
    TextTokenInput,
    LLMAttributionResult,
)

import os

os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

model_name = 'meta-llama/Llama-3.1-8B-Instruct'

def load_dataset(json_path: str = 'dataset/mitigation_data/false_consensus_mitigation.json'):
    with open(json_path, 'r') as file:
        question_all = json.load(file)
    return question_all

question_all = load_dataset()


def get_logprob_chat(input_messages, model_name: str, output_token: str):
    
    client = OpenAI(api_key="YOUR API KEY",base_url= "YOUR ENDPOINT URL")
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
                    print(f"Yes! target token '{output_token}' is in top20: {logprob_dict}")
                    res.append(logprob_dict[output_token])
                else:  # When the target token is not in the top 20 tokens, a value based on experience is taken. This can be optimized through the probability distribution of the vocabulary output by the large language model.
                    print(f"Oh no! target token '{output_token}' is not in top20: {logprob_dict}")
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


def featureAbalationSegmentChat(input_messages, model_name: str, target=None, base_segment=''):
   
    client = OpenAI(api_key="YOUR API KEY",base_url= "YOUR ENDPOINT URL")

    input_segment_length = len(input_messages)

    if target is None:
        response = client.chat.completions.create(
            model=model_name,
            messages=input_messages,
            max_tokens=1
        )
        target = response.choices[0].message.content

    attr = np.zeros((1, input_segment_length))
    ref = get_logprob_chat(input_messages, model_name, target)

    for i in range(input_segment_length):
        modified_messages = [dict(item) for item in input_messages]
        modified_messages[i]['content'] = base_segment
        attr[:, i] = ref - get_logprob_chat(modified_messages, model_name, target)

    seq_attr = np.sum(attr, axis=0)

    return LLMAttributionResult(torch.tensor(seq_attr), torch.tensor(attr),
                                [item['content'] for item in input_messages], [target],target=None)


def featureAbalationChat(input_messages, model_name: str, target=None, base_token=''):
    
    enc = tiktoken.encoding_for_model(model_name)
    # EXTRACT full_content, question, bias, option
    full_content = input_messages[1]['content']
    question_end_index = full_content.index('Most')
    question = full_content[:question_end_index].strip()
    rest_content = full_content[question_end_index:].strip()
    content_parts = rest_content.split('\n')
    mitigation = "Remain open to multiple perspectives. Just because a belief is widely held doesn't mean it's correct. Independently verify facts and consider a wide range of viewpoints."
    bias = content_parts[0].replace(mitigation, "").strip()
    option = content_parts[1].strip()

    #  question, bias, option encode to token id list
    question_token_ids = enc.encode(question)
    bias_token_ids = enc.encode(bias)
    mitigation_token_ids = enc.encode(mitigation)
    option_token_ids = enc.encode(option)

    #  question, bias, option transfer to token list
    question_tokens = [enc.decode([token_id]) for token_id in question_token_ids]
    bias_tokens = [enc.decode([token_id]) for token_id in bias_token_ids]
    mitigation_tokens = [enc.decode([token_id]) for token_id in mitigation_token_ids]
    option_tokens = [enc.decode([token_id]) for token_id in option_token_ids]

    #  token list
    input_token_id_list_list = [enc.encode(item['content']) for item in input_messages]
    input_token_list_list = []  # List of lists of tokens
    input_token_list = []  # Flattened list of tokens

    for token_id_list_partial in input_token_id_list_list:
        token_list_partial = [enc.decode([token_id]) for token_id in token_id_list_partial]
        input_token_list_list.append(token_list_partial)
        input_token_list += token_list_partial
    input_token_length = len(input_token_list)

   
    # initial
    attr = np.zeros((1, input_token_length))
    ref = get_logprob_chat(input_messages, model_name, target)
    cnt = 0

    segments = [question_tokens, bias_tokens, mitigation_tokens, option_tokens]


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
                modified_messages[1]['content'] = temp_prompt+bias+mitigation+option
            elif i==1:
                modified_messages[1]['content'] = question+temp_prompt+mitigation+option
            elif i==2:
                modified_messages[1]['content'] = question+bias+temp_prompt+option
            elif i==3:
                modified_messages[1]['content'] = question+bias+mitigation+temp_prompt
            # modified_messages[1]['content'] = temp_prompt
            attr[:, cnt] = ref - get_logprob_chat(modified_messages, model_name, target)
            cnt += 1

   
    seq_attr = np.sum(attr, axis=0)
    attr_segment = np.zeros((1, 4))  

    start = 0
    for i, tokens in enumerate(segments):
        end = start + len(tokens)
        attr_segment[0, i] = np.average(attr[0, start:end])
        start = end

    seq_attr_segment = np.sum(attr_segment, axis=0)

    return (LLMAttributionResult(torch.tensor(seq_attr), torch.tensor(attr), input_token_list, [target]),
            LLMAttributionResult(torch.tensor(seq_attr_segment), torch.tensor(attr_segment),
                                 ['question', 'bias', 'mitigation', 'option'], [target]))


def plot_new_visualization(AttributionResult: LLMAttributionResult, model_name: str, save: bool = False) -> Image:
    
    text_parts = []

    data = AttributionResult.seq_attr.cpu().numpy()

    n_inputs = len(AttributionResult.input_tokens)
    for i in range(n_inputs):

        if data[i] > 0:
            # Light green: HSV conversion: Hue=90 degrees, Value=90%, Saturation increases with data[i]
            hue = 90
            value = 0.9
            saturation = min(0.2 + 0.6 * data[i], 1)
            bg_color = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            bg_color_rgb = tuple(int(c * 255) for c in bg_color)
            changed_text_parts = f"{AttributionResult.input_tokens[i]}".replace('Ġ', ' ').replace('Ċ', '\n')
            if changed_text_parts == '\n\n':
                text_parts.append({
                    "text": changed_text_parts,
                    "bg_color": (255, 255, 255),
                })
            else:
                text_parts.append({
                    "text": changed_text_parts,
                    "bg_color": bg_color_rgb,
                })



        elif data[i] < 0:
            # Light yellow: HSV conversion: Hue=50 degrees, Value=100%, Saturation increases with data[i]
            hue = 60
            value = 1.0
            saturation = min(0.2 - 1 * data[i], 1)
            bg_color = colorsys.hsv_to_rgb(hue / 360, saturation, value)
            bg_color_rgb = tuple(int(c * 255) for c in bg_color)
            changed_text_parts = f"{AttributionResult.input_tokens[i]}".replace('Ġ', ' ').replace('Ċ', '\n')
            if changed_text_parts == '\n\n':
                text_parts.append({
                    "text": changed_text_parts,
                    "bg_color": (255, 255, 255),
                })
            else:
                text_parts.append({
                    "text": changed_text_parts,
                    "bg_color": bg_color_rgb,
                })
        else:
            changed_text_parts = f"{AttributionResult.input_tokens[i]}".replace('Ġ', ' ').replace('Ċ', '\n')
            text_parts.append({
                "text": changed_text_parts,
                "bg_color": (255, 255, 255),
            })

    if 'Llama' in model_name:
        image = create_colored_text_image(text_parts=text_parts[1:])
    elif 'gpt' in model_name:
        image = create_colored_text_image(text_parts=text_parts)
    if save:
        image.save("attribute_colored_text_image.png")
    else:
        plt.imshow(image)
        plt.axis('off')
        plt.show()
        return image


def create_colored_text_image(
        text_parts: list,
        font_path: str = "Arial.ttf",
        font_size: int = 40,
        background_color: str = "white",
        max_width: int = 785,
        line_spacing: int = 15
) -> Image:
    
    # Use the specified font
    font = ImageFont.truetype(font_path, font_size)

    # Initialize variables
    lines = []
    current_line = []
    current_width = 0

    # Calculate line breaks
    for part in text_parts:
        text = part["text"]
        bg_color = part["bg_color"]

        # Process word by word
        for word in text.split(' '):
            word_width = font.getbbox(word)[2]

            # Check if it exceeds the line width
            if current_width + word_width > max_width:
                lines.append(current_line)
                current_line = []
                current_width = 0

            # Add word to current line
            current_line.append({"text": word, "bg_color": bg_color})
            current_width += word_width + font.getbbox(' ')[2]  # Include space width

        # Check for line breaks
        if '\n' in text:
            lines.append(current_line)
            current_line = []
            current_width = 0

    # Add the last line
    if current_line:
        lines.append(current_line)

    # Calculate image height
    image_height = len(lines) * (font_size + line_spacing) + 20
    image = Image.new("RGBA", (max_width + 20, image_height), background_color)
    draw = ImageDraw.Draw(image)

    # Draw text and background line by line
    y = 10
    for line in lines:
        x = 10  # Starting x-coordinate for each line
        for part in line:
            text = part["text"]
            bg_color = part["bg_color"]

            # Get text width and height, then draw background rectangle
            text_width, text_height = font.getbbox(text)[2], font.getbbox(text)[3]
            draw.rectangle([(x, y), (x + text_width, y + text_height)], fill=bg_color)

            # Draw the text
            draw.text((x, y), text, font=font, fill="black")  # Default text color is black
            x += text_width + font.getbbox(' ')[2]  # Update x position, including space width
        y += font_size + line_spacing  # Update y position

    return image


def create_contribution_visualization(model_name: str, question_all: dict) -> Image:

    image = []
    instruction = "You are a medical Llama model tasked with making clinical decisions for research puposes only. Please choose one of the following options. Respond with only a single letter and nothing more."

    if "Llama" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if "gpt" in model_name:
        client = OpenAI(api_key="your api key",
                        base_url="")

    for data in question_all:

        question = data["question"]
        options = data["options"]
        options_string = ", ".join([f"{key}: {value}" for key, value in options.items()])
        output = f"{question}\n{options_string}"
        first_question = output + '\n'
        true_answer = data['answer_idx']
        bias_answer = data['bias_answer_index']


        messages = [{'role': 'system', 'content': instruction},{'role': 'user', 'content': first_question}]

        if 'gpt' in model_name:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1
            )
            target = response.choices[0].message.content

            if target == bias_answer:
                _,attr_res = featureAbalationChat(messages, model_name, target=target)
            else:
                continue

        elif 'Llama' in model_name:
            eval_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

            prompt = tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

            with torch.no_grad():
                generate_ids = model.generate(
                    inputs.input_ids,
                    # inputs.attention_mask,
                    max_new_tokens=1,
                    
                )

            target_llama = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[
                0][-1]
            # target_llama = bias_answer

            # if target_llama == true_answer:
            inp = TextTokenInput(
                eval_prompt,
                tokenizer,
                skip_tokens=[1],  
            )

            fa = FeatureAblation(model)
            llm_attr = LLMAttribution(fa, tokenizer)
            attr_res_temp = (llm_attr.attribute(inp, target=target_llama))
            attr = attr_res_temp.token_attr.cpu()
            # calcultate seq_attr and attr_segment
            attr_segment = np.zeros((1, 4))  # because there are segments：question, bias, mitigation, option

            full_content = messages[1]['content']
            question_end_index = full_content.index('Most')
            question = full_content[:question_end_index].strip()
            rest_content = full_content[question_end_index:].strip()
            content_parts = rest_content.split('\n')
            mitigation = "Remain open to multiple perspectives. Just because a belief is widely held doesn't mean it's correct. Independently verify facts and consider a wide range of viewpoints."
            bias = content_parts[0].replace(mitigation, "").strip()
            option = content_parts[1].strip()

            #  question, bias, option encode to token id list
            question_token_ids = tokenizer.encode(question)
            bias_token_ids = tokenizer.encode(bias)
            mitigation_token_ids = tokenizer.encode(mitigation)
            option_token_ids = tokenizer.encode(option)
            # question_token_ids = tokenizer(question, return_tensors="pt").input_ids

            #  question, bias, option transfer to  token list
            question_tokens = [tokenizer.decode([token_id],skip_special_tokens=True) for token_id in question_token_ids]
            bias_tokens = [tokenizer.decode([token_id],skip_special_tokens=True) for token_id in bias_token_ids]
            mitigation_tokens = [tokenizer.decode([token_id],skip_special_tokens=True) for token_id in mitigation_token_ids]
            option_tokens = [tokenizer.decode([token_id],skip_special_tokens=True) for token_id in option_token_ids]

            segments = [question_tokens, bias_tokens, mitigation_tokens, option_tokens]

            start = 67
            for i, tokens in enumerate(segments):
                end = start + len(tokens)-1       
                attr_segment[0, i] = np.average(attr[0, start:end])
                start = end

            seq_attr_segment = np.sum(attr_segment, axis=0)
            attr_res = LLMAttributionResult(torch.tensor(seq_attr_segment), torch.tensor(attr_segment),
                                 ['question', 'bias', 'mitigation', 'option'], [target_llama])
            
        image.append(attr_res.plot_token_attr(show=True))


    return image

create_contribution_visualization(model_name=model_name,question_all=question_all)