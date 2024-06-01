import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def topKaccuracy(model, data_loader, topk=(1,)):
    device = next(model.parameters()).device
    with torch.no_grad():
        total = 0
        corrects = {k: 0 for k in topk}
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.topk(outputs, max(topk), dim=1)
            total += labels.size(0)
            for k in topk:
                corrects[k] += torch.sum(torch.any(predicted[:, :k] == labels.view(-1, 1), dim=1)).item()
        accuracies = [(corrects[k] / total)  for k in topk]

    return accuracies




# Chuyển đổi danh sách thành numpy arrays
y_true = np.array(y_true_list)
y_pred = np.array(y_pred_list)
# Hàm tính precision, recall và F1-score
def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    return precision, recall, f1

# Hàm vẽ confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, linewidths=.5, linecolor='gray')  # Thêm viền và màu viền
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == "__main__":
    # Lặp qua các batch từ DataLoader
    val_dataloader = DataLoader(
            train_dataset,
            batch_size=len(train_df),
        )

    y_true_list = []
    y_pred_list = []

    for batch_data, batch_labels in val_dataloader:
        with torch.no_grad():
            output = model(batch_data)  # Forward pass để dự đoán nhãn
            _, predicted = torch.max(output, 1)  # Lấy nhãn dự đoán

        # Thu thập nhãn thực và dự đoán từ batch hiện tại
        y_true_list.extend(batch_labels.numpy())  # Chuyển tensor sang numpy array và mở rộng list
        y_pred_list.extend(predicted.numpy())  # Chuyển tensor sang numpy array và mở rộng list
    # Example usage:
    precision, recall, f1 = calculate_metrics(y_true, y_pred)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-score:", f1)

    with open('dataset/sign_to_prediction_index_map.json', 'r', encoding='utf-8') as json_file:
        label_mapping = json.load(json_file)
    classes =[key for key, _ in label_mapping.items()] # Tên của các lớp
    plot_confusion_matrix(y_true, y_pred, classes)