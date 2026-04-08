import json


file_name = 'dataset/bias_confirmation_dev.json'
with open(file_name, 'r', encoding='utf-8') as file:
    data = json.load(file)

for item in data:
    item['question'] += (
        "Encourage a comprehensive approach to diagnosis. Actively seek out information that both confirms and disconfirms your initial hypothesis to ensure a balanced view."
    )

output_file_name = 'dataset/confirmation_mitigation.json'
with open(output_file_name, 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print(f"Updated JSON has been saved to {output_file_name}")