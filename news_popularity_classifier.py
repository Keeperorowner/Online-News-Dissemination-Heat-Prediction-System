import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保结果可重现
np.random.seed(42)
torch.manual_seed(42)

class NewsPopularityClassifier(nn.Module):
    """
    基于前馈神经网络的新闻热度分类器
    使用Softmax回归进行四分类预测
    """
    def __init__(self, input_size, hidden_sizes, num_classes=4, dropout_rate=0.5):
        super(NewsPopularityClassifier, self).__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        # 添加隐藏层
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        # 输出层
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

def load_and_preprocess_data(file_path):
    """
    加载并预处理数据
    """
    # 读取数据
    df = pd.read_csv(file_path)
    
    # 查看列名并清理列名中的空格
    df.columns = df.columns.str.strip()
    print("Columns in dataset:", df.columns.tolist())
    
    # 删除非特征列
    columns_to_drop = [col for col in ['url', 'timedelta'] if col in df.columns]
    df = df.drop(columns_to_drop, axis=1)
    
    # 分离特征和目标变量
    X = df.drop('shares', axis=1)
    y = df['shares']
    
    # 将分享数转换为四个类别:
    # 冷启动 (0-1400 shares)
    # 一般传播 (1400-2000 shares) 
    # 高传播 (2000-5000 shares)
    # 爆火 (5000+ shares)
    y_class = pd.cut(y, bins=[0, 1400, 2000, 5000, float('inf')], 
                     labels=[0, 1, 2, 3], right=False)
    y_class = y_class.astype(int)
    
    return X.values, y_class.values

def train_model(model, X_train, y_train, X_val, y_val, 
                num_epochs=100, learning_rate=0.001, batch_size=64):
    """
    训练模型
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train), torch.LongTensor(y_train))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # 训练模型
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(torch.FloatTensor(X_val))
            val_loss = criterion(val_outputs, torch.LongTensor(y_val)).item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_accuracy = accuracy_score(y_val, val_predicted.numpy())
        
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        # 每10个epoch打印一次信息
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}, '
                  f'Train Acc: {train_accuracies[-1]:.4f}, '
                  f'Val Acc: {val_accuracies[-1]:.4f}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    """
    model.eval()
    with torch.no_grad():
        outputs = model(torch.FloatTensor(X_test))
        _, predicted = torch.max(outputs.data, 1)
        
        accuracy = accuracy_score(y_test, predicted.numpy())
        print(f'Test Accuracy: {accuracy:.4f}')
        
        print('\nClassification Report:')
        print(classification_report(y_test, predicted.numpy(), 
                                  target_names=['Cold Start', 'General', 'High', 'Viral']))
        
        # 绘制混淆矩阵
        cm = confusion_matrix(y_test, predicted.numpy())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Cold Start', 'General', 'High', 'Viral'],
                    yticklabels=['Cold Start', 'General', 'High', 'Viral'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
    return accuracy, predicted.numpy()

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    绘制训练历史
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # 准确率曲线
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    """
    主函数
    """
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('./OnlineNewsPopularity/OnlineNewsPopularity.csv')
    
    # 划分数据集 (70% 训练, 15% 验证, 15% 测试)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Validation set size: {X_val.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    print(f"Number of features: {X_train.shape[1]}")
    
    # 统计各类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nTraining set class distribution:")
    class_names = ['Cold Start', 'General', 'High', 'Viral']
    for i, (cls, count) in enumerate(zip(unique, counts)):
        print(f"  {class_names[cls]} ({cls}): {count} samples")
    
    # 创建模型
    input_size = X_train.shape[1]
    hidden_sizes = [256, 128, 64]  # 隐藏层大小
    model = NewsPopularityClassifier(input_size, hidden_sizes, num_classes=4, dropout_rate=0.3)
    
    print(f"\nModel architecture:")
    print(model)
    
    # 训练模型
    print("\nTraining model...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, X_train, y_train, X_val, y_val, 
        num_epochs=100, learning_rate=0.001, batch_size=128)
    
    # 绘制训练历史
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 评估模型
    print("\nEvaluating model on test set...")
    test_accuracy, predictions = evaluate_model(model, X_test, y_test)
    
    print(f"\nFinal Results:")
    print(f"Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()