import json

# Load the JSON file
with open(r'C:\Users\Asus TUF\Documents\code\TA\simsiam\simsiam\TESTING\MyMethod\performance\MyMethod 20250615_134218_results.json', 'r') as f:
    data = json.load(f)

# Extract the performance dictionary
performance = data.get('performance', {})

for metric, value in performance.items():
    formatted_value = f"{value:.6f}".replace('.', ',')
    print(f"{formatted_value}")
