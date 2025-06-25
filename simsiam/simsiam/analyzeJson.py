import json

# Load the JSON file
with open(r'C:\Users\PC\Documents\code\TA\tugas-akhir-model-deteksi-tumpahan-minyakl\simsiam\simsiam\TESTING\new_mymethod\performance\MyMethod 20250623_202112_results.json', 'r') as f:
    data = json.load(f)

predictions = data.get('prediction')
total_dataset = data.get("performance")

print("=====AUC=====")
for i in range(10):
    pred = predictions[i]
    print(f"{pred['AUC']:.4f}".replace('.', ','))

print(f"{total_dataset['AUC']:.4f}".replace('.', ','))

print("=====precision=====")
for i in range(10):
    pred = predictions[i]
    print(f"{pred['precision']:.4f}".replace('.', ','))

print(f"{total_dataset['precision']:.4f}".replace('.', ','))

print("=====recall=====")
for i in range(10):
    pred = predictions[i]
    print(f"{pred['recall']:.4f}".replace('.', ','))

print(f"{total_dataset['recall']:.4f}".replace('.', ','))

print("=====F1 Score=====")
for i in range(10):
    pred = predictions[i]
    print(f"{pred['F1 Score']:.4f}".replace('.', ','))

print(f"{total_dataset['F1 Score']:.4f}".replace('.', ','))

print("=====OA=====")
for i in range(10):
    pred = predictions[i]
    print(f"{pred['OA']:.4f}".replace('.', ','))

print(f"{total_dataset['OA']:.4f}".replace('.', ','))

print("=====AA=====")
for i in range(10):
    pred = predictions[i]
    print(f"{pred['AA']:.4f}".replace('.', ','))

print(f"{total_dataset['AA']:.4f}".replace('.', ','))