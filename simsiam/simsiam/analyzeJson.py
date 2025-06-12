import json

# Load the JSON file
with open(r'C:\Users\Asus TUF\Documents\code\TA\simsiam\simsiam\TESTING\MyMethod\performance\MyMethod 20250611_120532_results.json', 'r') as f:
    data = json.load(f)

# Extract the performance dictionary
performance = data.get('prediction')

for i in range(10):
    acc = (performance[i]['correct_total']/performance[i]['total'])
    print(f"{acc:.4f}")
