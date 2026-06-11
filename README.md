# 基于随机森林的 WDBC 乳腺癌数据集分类

机器学习大作业选题二第二部分，使用随机森林（Random Forest）对 WDBC（Wisconsin Diagnostic Breast Cancer）数据集进行二分类预测。

## 项目内容

| 文件 | 说明 |
|------|------|
| `src/breast_cancer_rf.ipynb` | 主实验 Notebook，包含完整的数据分析、模型训练与评估流程 |
| `src/random_forest_pipeline.py` | 纯 Python 算法脚本，可直接运行 |
| `breast_cancer_rf.html` | Notebook 导出的 HTML 报告 |
| `data/wdbc.data` | WDBC 原始数据集 |
| `data/wdbc.names` | 数据集字段说明 |
| `figures/` | 所有分析图表（混淆矩阵、ROC 曲线、特征重要性等） |
| `tests/test_pipeline.py` | 流水线功能测试 |
| `requirements.txt` | 项目依赖列表 |

## 数据集

- **来源**：UCI Machine Learning Repository — WDBC
- **样本数**：569 条
- **特征数**：30 个数值特征（由细胞核图像的 10 个属性分别计算均值、标准差、最大值得到）
- **标签**：`M`（恶性 Malignant）/ `B`（良性 Benign），编码为 `1` / `0`
- **任务**：二分类

## 环境搭建

```bash
# 创建虚拟环境（推荐）
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

## 运行 Notebook

```bash
jupyter notebook src/breast_cancer_rf.ipynb
```

或直接运行算法脚本：

```bash
python src/random_forest_pipeline.py
```

## 测试方法

```bash
python -m pytest tests/
```

## 使用的分类方法

| 方法 | 说明 |
|------|------|
| **Random Forest（随机森林）** | 本作业实现方法，集成学习，基于多棵决策树投票 |

> 同组其他成员使用 ANN（人工神经网络）方法，见各自仓库。

## 训练条件与模型参数

| 条件 | 设置 |
|------|------|
| 数据划分 | 训练集 70% / 测试集 30%，随机种子 42 |
| 标签编码 | `M` → `1`（恶性），`B` → `0`（良性） |
| 特征处理 | 标准化（StandardScaler） |
| 评价指标 | Accuracy、Precision、Recall、F1-Score、ROC-AUC |

**随机森林关键参数：**

| 参数 | 值 |
|------|-----|
| `n_estimators` | 100 |
| `max_depth` | None（不限制） |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `random_state` | 42 |

## 当前结果

| 指标 | 随机森林 |
|------|----------|
| **Accuracy** | ~96% |
| **Precision** | ~97% |
| **Recall** | ~95% |
| **F1-Score** | ~96% |
| **ROC-AUC** | ~99% |

> 结果来自 `src/breast_cancer_rf.ipynb` 最后一次运行，具体数值以 Notebook 输出为准。

## 参考与借鉴

- [UCI WDBC Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
- [scikit-learn RandomForestClassifier 文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- 课程：合肥工业大学《机器学习》2026 年春

## 作者

- 谭鹏飞（115 2023218584）
- 合肥工业大学
