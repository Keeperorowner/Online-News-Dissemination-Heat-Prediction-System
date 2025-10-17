# 在线新闻传播热度预测系统

本项目基于前馈神经网络（FNN）和Softmax回归实现在线新闻传播热度的四分类预测。使用UCI的Online News Popularity Dataset（在线新闻流行度数据集）进行训练和测试。

## 项目概述

该项目旨在预测在线新闻的传播热度，将新闻分为四个类别：
1. 冷启动 (Cold Start) - 0-1400次分享
2. 一般传播 (General) - 1400-2000次分享
3. 高传播 (High) - 2000-5000次分享
4. 爆火 (Viral) - 5000+次分享

## 项目结构

```
.
├── news_popularity_classifier.py  # 主要的模型实现代码
├── README.md                      # 项目说明文档
├── requirements.txt               # 项目依赖列表
└── .gitignore                    # Git忽略文件配置
```

## 数据集

使用UCI在线新闻流行度数据集：
- 包含39,797条新闻记录
- 每条记录有58个特征
- 目标变量为新闻在社交媒体上的分享次数(shares)

数据集下载地址: https://archive.ics.uci.edu/dataset/332/online+news+popularity

## 环境要求

- Python 3.7+
- pandas
- numpy
- torch (PyTorch)
- scikit-learn
- matplotlib
- seaborn

## 安装步骤

1. 克隆或下载本项目到本地
2. 安装所需依赖:
   ```
   pip install -r requirements.txt
   ```

3. 下载数据集并解压到项目目录中，确保目录结构如下:
   ```
   .
   ├── OnlineNewsPopularity/
   │   ├── OnlineNewsPopularity.csv
   │   └── OnlineNewsPopularity.names
   ├── news_popularity_classifier.py
   └── ...
   ```

## 运行项目

执行以下命令运行模型训练和评估:
```
python news_popularity_classifier.py
```

## 模型架构

采用前馈神经网络（FNN）结合Softmax回归：
- 输入层：58个特征节点
- 隐藏层：256、128、64个节点
- 输出层：4个节点（对应4个分类）
- 激活函数：ReLU
- 正则化：Dropout (0.3)
- 优化器：Adam
- 损失函数：交叉熵损失

## 训练过程

1. 数据预处理：
   - 清理列名空格
   - 删除非特征列（url, timedelta）
   - 将分享数转换为四分类标签

2. 数据划分：
   - 训练集：70% (27,750样本)
   - 验证集：15% (5,947样本)
   - 测试集：15% (5,947样本)

3. 特征标准化：
   - 使用StandardScaler进行标准化处理

4. 模型训练：
   - 批次大小：128
   - 学习率：0.001
   - 训练轮数：100

## 结果分析

模型在测试集上达到了约48%的准确率。分类报告显示：
- 冷启动类别预测效果最好（精确率0.55，召回率0.81）
- 其他类别预测效果有待提升

## 过拟合控制

采用以下方法控制过拟合：
1. Dropout正则化（dropout_rate=0.3）
2. L2正则化（weight_decay=1e-5）
3. 合理的网络结构设计

## 改进建议

1. 特征工程优化
2. 调整类别不平衡问题
3. 尝试其他模型架构
4. 超参数调优