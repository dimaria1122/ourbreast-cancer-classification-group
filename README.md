# 乳腺癌分类预测 — 多种机器学习方法对比

机器学习大作业，基于 Kaggle 乳腺癌数据集，对比六种分类算法的表现：KNN、Naive Bayes、SVM、Logistic Regression、Decision Tree、Random Forest。

## 使用的库

```bash
pip install numpy pandas seaborn matplotlib scikit-learn
```

## 分类方法

| 方法 | 说明 |
|------|------|
| KNN | K 最近邻 |
| Naive Bayes | 高斯朴素贝叶斯 |
| SVM | 支持向量机 |
| Logistic Regression | 逻辑回归（表现最优） |
| Decision Tree | 决策树 |
| Random Forest | 随机森林 |

## 运行

```bash
jupyter notebook breast_cancer_ann.ipynb
```

## 注意

Notebook 中数据集路径为 Kaggle 环境路径 `../input/datasets/organizations/uciml/breast-cancer-wisconsin-data/data.csv`，本地运行需修改为实际数据集路径。
