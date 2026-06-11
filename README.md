# 基于sklearn的乳腺癌数据集四种分类方法和可视化

本仓库是我们小组机器学习大作业中的 scikit-learn 分类实验部分。本部分实验基于 Kaggle 的 Breast Cancer Wisconsin (Diagnostic) Data Set，对乳腺癌诊断结果进行二分类，并在 Jupyter Notebook 中完成数据清洗、模型训练、结果比较和可视化。

说明：小组内其他成员可能会使用随机森林、支持向量机、朴素贝叶斯、神经网络等不同方法，或者使用不同软件包和不同模型参数。本仓库只说明我负责的 scikit-learn 实验部分（mac上实现的）。

本部分主要完成：

- 使用本地 Kaggle 数据 `data.csv`
- 使用 scikit-learn 尝试 4 种分类方法
- 对分类结果进行指标比较
- 对数据分布、混淆矩阵、ROC 曲线和特征重要性进行可视化

## 项目内容

- 数据集：`data.csv`
- 实验 Notebook：`breast_cancer_classification.ipynb`
- 工程源代码：`src/breast_cancer_pipeline.py`
- 依赖文件：`requirements.txt`
- 测试代码：`tests/test_pipeline.py`

## 数据集

Kaggle 地址：

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

数据共有 569 条样本。目标列为 `diagnosis`：

- `M`：恶性，编码为 `1`
- `B`：良性，编码为 `0`

建模时删除 `id` 和空列，保留 30 个数值特征。

## 参考与借鉴

调研时参考了 Kaggle 和公开 GitHub Notebook 的常见流程，包括删除无关列、标签编码、标准化、模型比较、混淆矩阵、ROC 曲线和特征重要性可视化。本实验没有直接复制他人的 Notebook，而是使用本地 `data.csv` 重新实现。

- Kaggle 数据集代码页：https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/code
- Kaggle: Predicting Breast Cancer with Random Forest (~95%)：https://www.kaggle.com/code/pratikkgandhi/predicting-breast-cancer-with-random-forest-95
- GitHub SVC 参考：https://github.com/mariyagolchin/Breast_Cancer_Classification-sklearn_library_Breast-cancer-wisconsin-dataset
- GitHub Random Forest 参考：https://github.com/shirinsamani/Diagnostic-Breast-Cancer-Wisconsin-Dataset

## 环境搭建

在 macOS 上建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 运行 Notebook

启动 Jupyter：

```bash
jupyter notebook breast_cancer_classification.ipynb
```

也可以命令行重新执行整个 Notebook：

```bash
jupyter nbconvert --to notebook --execute breast_cancer_classification.ipynb --output breast_cancer_classification.ipynb
```

## 测试

测试不是作业运行的必要步骤，但可以用来确认数据清洗和模型训练流程是否正常：

```bash
pytest
```

测试覆盖：

- Kaggle CSV 清洗后为 569 行、30 个特征
- 标签稳定编码为 `{0, 1}`
- 模型列表包含 4 个分类器
- 4 个模型均能训练并输出 Accuracy、Precision、Recall、F1、ROC-AUC 和混淆矩阵

## 本部分使用的分类方法

本实验使用 scikit-learn 中的 4 种分类模型。为避免和小组其他成员的实现混淆，下面列出本部分实际使用的 sklearn 类名：

| 模型名称            | sklearn 类名                                | 说明                             |
| ------------------- | ------------------------------------------- | -------------------------------- |
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | 逻辑回归，作为线性基线模型       |
| SVM                 | `sklearn.svm.SVC`                         | 支持向量机，适合中小规模分类任务 |
| Random Forest       | `sklearn.ensemble.RandomForestClassifier` | 随机森林，能够输出特征重要性     |
| KNN                 | `sklearn.neighbors.KNeighborsClassifier`  | K 近邻，基于样本距离进行分类     |

其中 Logistic Regression、SVM 和 KNN 对特征尺度敏感，因此使用 `StandardScaler` 做标准化；Random Forest 是树模型，对特征尺度不敏感，因此直接使用原始数值特征训练。

## 训练条件与模型参数

我本实验明确记录训练条件如下：

- 数据划分：`train_test_split(test_size=0.2, random_state=42, stratify=y)`
- 标签编码：`B -> 0`，`M -> 1`
- 特征处理：删除 `id` 和空列，保留 30 个数值特征
- 标准化处理：Logistic Regression、SVM、KNN 使用 `StandardScaler`
- 评价指标：Accuracy、Precision、Recall、F1、ROC-AUC、混淆矩阵

本实验中 4 个模型的主要参数设置如下：

| 模型名称            | 主要参数                                                               |
| ------------------- | ---------------------------------------------------------------------- |
| Logistic Regression | `solver="liblinear"`, `max_iter=1000`, `random_state=42`         |
| SVM                 | `kernel="rbf"`, `probability=True`, `random_state=42`            |
| Random Forest       | `n_estimators=300`, `class_weight="balanced"`, `random_state=42` |
| KNN                 | `n_neighbors=5`                                                      |

## 当前结果

使用 `test_size=0.2`、`random_state=42`、分层抽样划分测试集，当前测试集结果如下：

| Model               | Accuracy | Precision | Recall |     F1 | ROC-AUC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   0.9737 |    0.9756 | 0.9524 | 0.9639 |  0.9960 |
| SVM                 |   0.9737 |    1.0000 | 0.9286 | 0.9630 |  0.9947 |
| Random Forest       |   0.9649 |    1.0000 | 0.9048 | 0.9500 |  0.9970 |
| KNN                 |   0.9561 |    0.9744 | 0.9048 | 0.9383 |  0.9823 |

从结果看，四种模型都取得了较好的分类效果。Logistic Regression 和 SVM 的 Accuracy 较高，Random Forest 的 ROC-AUC 最高，并且可以提供特征重要性用于解释。
