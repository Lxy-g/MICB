import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json  

file_list = ['no_attack_file.json', 'file1.json', 'file2.json', 'file3.json']  


no_attack_average_frequency = None
attack_average_frequencies = []


for idx, file_name in enumerate(tqdm(file_list)):

    with open(file_name, 'r') as file:
        data = json.load(file)  

    sign_changes_list = []


    if idx == 0:  
        for no_attack_sample in data['no_attack_data']:
            correct_without_attack = np.array(no_attack_sample['layer_confidences']['Correct'])
            incorrect_without_attack = np.array(no_attack_sample['layer_confidences']['Incorrect'])

            # Compute Pr(correct) - Pr(incorrect)
            diff_without_attack = correct_without_attack - incorrect_without_attack


            sign_changes = np.sum(np.diff(np.sign(diff_without_attack)) != 0)


            sign_changes_list.append(sign_changes)


        no_attack_average_frequency = np.mean(sign_changes_list)
    else:  
       
        for attack_sample in data['attack_data']:
            correct_with_attack = np.array(attack_sample['layer_confidences']['Correct'])
            incorrect_with_attack = np.array(attack_sample['layer_confidences']['Incorrect'])

            # Compute Pr(correct) - Pr(incorrect)
            diff_with_attack = correct_with_attack - incorrect_with_attack

            sign_changes = np.sum(np.diff(np.sign(diff_with_attack)) != 0)

            sign_changes_list.append(sign_changes)

        attack_average_frequency = np.mean(sign_changes_list)
        attack_average_frequencies.append(attack_average_frequency)


plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(file_list) - 1)


plt.bar(index + bar_width, attack_average_frequencies, bar_width, label='Attack Data')


plt.bar(index, [no_attack_average_frequency] * len(index), bar_width, label='No Attack Data')

plt.xlabel('Files')
plt.ylabel('Average Frequency of Sign Changes')
plt.title('Comparison of Average Frequency of Sign Changes Across Files')
plt.xticks(index + bar_width / 2, [f'File {i+1}' for i in range(len(file_list) - 1)])
plt.legend()
plt.show()