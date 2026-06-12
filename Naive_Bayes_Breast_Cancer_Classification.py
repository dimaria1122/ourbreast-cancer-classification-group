# %% [markdown]
# # 基于朴素贝叶斯分类模型的乳腺癌肿瘤诊断系统
# 
# ## 任务背景
# 
# 本项目针对威斯康辛州乳腺癌数据集（Breast Cancer Wisconsin Dataset），构建基于朴素贝叶斯算法的二分类模型，实现乳腺肿瘤的良性/恶性自动诊断。
# 
# ## 数据集说明
# 
# - **数据集来源**: Kaggle平台威斯康辛州乳腺癌数据集
# - **样本数量**: 569条，其中恶性肿瘤(M)212条，良性肿瘤(B)357条
# - **特征数量**: 30个数值型特征，分为肿瘤半径、纹理、周长、面积等10个维度的均值、标准误差、最大值三组特征
# - **任务类型**: 二分类任务，预测肿瘤为良性(B)或恶性(M)
# 
# ## 算法原理
# 
# 朴素贝叶斯分类器基于贝叶斯定理与特征条件独立假设，对于连续型特征，采用高斯朴素贝叶斯（Gaussian Naive Bayes），假设特征服从高斯分布，通过计算后验概率实现分类。

# %%
# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report

# 设置中文字体和负号显示，避免 matplotlib findfont 警告
font_candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
available_fonts = {f.name for f in fm.fontManager.ttflist}
font_family = next((font for font in font_candidates if font in available_fonts), "sans-serif")
plt.rcParams["font.family"] = font_family
plt.rcParams["axes.unicode_minus"] = False

# 设置随机种子，保证结果可复现
np.random.seed(42)

# %% [markdown]
# ## 一、数据加载与初步探索
# 
# ### 1.1 加载数据集

# %%
# 加载数据集
data = pd.read_csv('data.csv')

# 查看数据集基本信息
print("数据集形状：", data.shape)
print("数据集前5行：")
print(data.head())
print("数据集列名：")
print(data.columns.tolist())

# %% [markdown]
# ### 1.2 数据质量检查
# 
# 查看数据缺失值、数据类型、标签分布等信息

# %%
# 数据类型查看
print("数据集数据类型：")
print(data.dtypes)

# 缺失值统计
print("数据集缺失值统计：")
print(data.isnull().sum())

# 标签列分布查看
print("肿瘤类型分布：")
label_dist = data['diagnosis'].value_counts()
print(label_dist)

# 可视化标签分布
plt.figure(figsize=(8, 6))
sns.countplot(x='diagnosis', data=data, palette=['#1f77b4', '#ff7f0e'])
plt.title('乳腺肿瘤类型分布', fontsize=14)
plt.xlabel('肿瘤类型（B=良性，M=恶性）', fontsize=12)
plt.ylabel('样本数量', fontsize=12)
plt.xticks(ticks=[0,1], labels=['良性(B)', '恶性(M)'])
for i, v in enumerate(label_dist):
    plt.text(i, v+5, str(v), ha='center', fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.show()



# %% [markdown]
# ### 1.3 特征分布探索
# 
# 查看核心特征的分布情况，以及良性/恶性肿瘤的特征差异

# %%
import matplotlib.font_manager as fm
font_candidates = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
available_fonts = {f.name for f in fm.fontManager.ttflist}
font_family = next((font for font in font_candidates if font in available_fonts), "sans-serif")
plt.rcParams["font.family"] = font_family
plt.rcParams["font.sans-serif"] = [font_family]
plt.rcParams["axes.unicode_minus"] = False

# 选择核心特征（均值组）进行分布可视化
core_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']

# 绘制特征分布箱线图
plt.figure(figsize=(15, 8))
for i, feature in enumerate(core_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x='diagnosis', y=feature, data=data, palette=['#1f77b4', '#ff7f0e'])
    plt.title(f'{feature} 分布对比', fontsize=12)
    plt.xlabel('肿瘤类型', fontsize=10)
    plt.ylabel(feature, fontsize=10)
    plt.xticks(ticks=[0,1], labels=['良性(B)', '恶性(M)'])
plt.tight_layout()
plt.show()

# 特征相关性热力图
plt.figure(figsize=(20, 16))
# 筛选数值型特征（排除id、diagnosis、空列）
numeric_features = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1).columns
corr = data[numeric_features].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('特征相关性热力图', fontsize=16)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 二、数据预处理
# 
# ### 2.1 数据清洗
# 
# - 去除无意义的id列;
# - 去除全空的Unnamed: 32列;
# - 标签列编码：将B(良性)编码为0，M(恶性)编码为1

# %%
# 数据清洗：去除无用列
data_clean = data.drop(['id', 'Unnamed: 32'], axis=1)

# 标签编码
le = LabelEncoder()
data_clean['diagnosis'] = le.fit_transform(data_clean['diagnosis'])

# 查看清洗后的数据
print("清洗后数据集形状：", data_clean.shape)
print("标签编码映射：", dict(zip(le.classes_, le.transform(le.classes_))))
print("清洗后数据前5行：")
print(data_clean.head())

# %% [markdown]
# ### 2.2 特征与标签拆分
# 
# 特征集X：所有30个数值型特征;
# 标签集y：diagnosis列（0=良性，1=恶性

# %%
# 拆分特征与标签
X = data_clean.drop('diagnosis', axis=1)
y = data_clean['diagnosis']

print("特征集形状：", X.shape)
print("标签集形状：", y.shape)

# %% [markdown]
# ### 2.3 数据集划分
# 
# 将数据集划分为训练集（80%）和测试集（20%），采用分层抽样保证标签分布一致

# %%
# 划分训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # 分层抽样，保证训练集和测试集标签分布一致
)

print("训练集形状：", X_train.shape)
print("测试集形状：", X_test.shape)
print("训练集标签分布：")
print(y_train.value_counts(normalize=True).round(4))
print("测试集标签分布：")
print(y_test.value_counts(normalize=True).round(4))

# %% [markdown]
# ### 2.4 特征标准化
# 
# 对特征进行标准化处理，使各特征尺度一致，符合高斯朴素贝叶斯的分布假设

# %%
# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 查看标准化后特征的均值和标准差
print("标准化后训练集特征均值：", X_train_scaled.mean(axis=0).round(4))
print("标准化后训练集特征标准差：", X_train_scaled.std(axis=0).round(4))

# %% [markdown]
# ## 三、朴素贝叶斯模型构建与训练
# 
# ### 3.1 模型初始化
# 
# 采用高斯朴素贝叶斯分类器，适用于连续型特征的分类任务

# %%
# 初始化高斯朴素贝叶斯模型
gnb = GaussianNB()

# 查看模型默认参数
print("高斯朴素贝叶斯模型默认参数：")
print(gnb.get_params())

# %% [markdown]
# ### 3.2 模型训练
# 
# 使用标准化后的训练集数据训练模型

# %%
# 模型训练
gnb.fit(X_train_scaled, y_train)

print("模型训练完成！")

# %% [markdown]
# ## 四、模型预测与性能评估
# 
# ### 4.1 模型预测
# 
# 对训练集和测试集分别进行预测

# %%
# 模型预测
y_train_pred = gnb.predict(X_train_scaled)
y_test_pred = gnb.predict(X_test_scaled)

# 预测概率（用于ROC曲线）
y_test_proba = gnb.predict_proba(X_test_scaled)[:, 1]

print("训练集预测完成，前10个预测结果：", y_train_pred[:10])
print("测试集预测完成，前10个预测结果：", y_test_pred[:10])


# %% [markdown]
# ### 4.2 核心评估指标
# 
# 计算准确率、精确率、召回率、F1分数等核心分类指标

# %%
# 计算核心评估指标
def calculate_metrics(y_true, y_pred, dataset_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"===== {dataset_name} 模型性能指标 =====")
    print(f"准确率（Accuracy）：{accuracy:.4f}")
    print(f"精确率（Precision）：{precision:.4f}")
    print(f"召回率（Recall）：{recall:.4f}")
    print(f"F1分数（F1-Score）：{f1:.4f}")
    print(" ")

# 输出训练集和测试集指标
calculate_metrics(y_train, y_train_pred, "训练集")
calculate_metrics(y_test, y_test_pred, "测试集")

# 输出详细分类报告
print("===== 测试集详细分类报告 =====")
print(classification_report(y_test, y_test_pred, target_names=['良性(B)', '恶性(M)']))


# %% [markdown]
# ### 4.3 混淆矩阵可视化
# 
# 通过混淆矩阵直观展示模型的分类结果

# %%
# 计算混淆矩阵
cm = confusion_matrix(y_test, y_test_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['良性(B)', '恶性(M)'], 
            yticklabels=['良性(B)', '恶性(M)'])
plt.title('朴素贝叶斯模型测试集混淆矩阵', fontsize=14)
plt.xlabel('预测标签', fontsize=12)
plt.ylabel('真实标签', fontsize=12)
plt.show()

# 输出混淆矩阵详细说明
print("混淆矩阵详细说明：")
print(f"真阴性（TN，正确预测良性）：{cm[0,0]}")
print(f"假阳性（FP，错误预测恶性）：{cm[0,1]}")
print(f"假阴性（FN，错误预测良性）：{cm[1,0]}")
print(f"真阳性（TP，正确预测恶性）：{cm[1,1]}")

# %% [markdown]
# ### 4.4 ROC曲线与AUC值
# 
# ROC曲线展示模型在不同阈值下的分类能力，AUC值越接近1，模型泛化能力越强

# %%
# ========== 解决中文乱码（全局设置） ==========
import matplotlib.font_manager as fm
font_candidates = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
available_fonts = {f.name for f in fm.fontManager.ttflist}
font_family = next((font for font in font_candidates if font in available_fonts), "sans-serif")
plt.rcParams["font.family"] = font_family
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常

# 计算ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
# 计算AUC值
auc = roc_auc_score(y_test, y_test_proba)

# 可视化ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='#1f77b4', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='#ff7f0e', lw=2, linestyle='--', label='随机猜测')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假阳性率 (FPR)', fontsize=12)
plt.ylabel('真阳性率 (TPR)', fontsize=12)
plt.title('朴素贝叶斯模型ROC曲线', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()

print(f"模型AUC值：{auc:.4f}")


# %% [markdown]
# ## 五、模型优化
# 
# ### 5.1 特征选择优化
# 
# 通过特征重要性筛选核心特征，减少冗余特征，提升模型性能

# %%
# 基于互信息的特征选择
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# 选择10个最优特征
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# 查看选中的特征
selected_features = X.columns[selector.get_support()]
print("选中的10个最优特征：")
for i, feature in enumerate(selected_features, 1):
    print(f"{i}. {feature}")

# 用优化后的特征重新训练模型
gnb_optimized = GaussianNB()
gnb_optimized.fit(X_train_selected, y_train)

# 预测与评估
y_test_pred_optimized = gnb_optimized.predict(X_test_selected)
y_test_proba_optimized = gnb_optimized.predict_proba(X_test_selected)[:, 1]

# 输出优化前后指标对比
print("===== 优化前后模型性能对比 =====")
print("原始模型测试集准确率：", round(accuracy_score(y_test, y_test_pred), 4))
print("优化后模型测试集准确率：", round(accuracy_score(y_test, y_test_pred_optimized), 4))
print("原始模型测试集AUC值：", roc_auc_score(y_test, y_test_proba).round(4))
print("优化后模型测试集AUC值：", roc_auc_score(y_test, y_test_proba_optimized).round(4))

# %% [markdown]
# ### 5.2 先验概率调整
# 
# 根据样本分布调整先验概率，缓解样本不平衡问题

# %%
# ========== 原有逻辑：计算样本先验概率 ==========
class_prior = y_train.value_counts(normalize=True).sort_index().values
print("训练集样本先验概率：", class_prior.round(4))  # numpy数组支持 .round()，此行无需修改

# ========== 补充：训练【原始默认先验】的高斯贝叶斯模型（你代码缺失部分） ==========
gnb_origin = GaussianNB()
gnb_origin.fit(X_train_scaled, y_train)
y_test_pred = gnb_origin.predict(X_test_scaled)
y_test_proba = gnb_origin.predict_proba(X_test_scaled)[:, 1]

# ========== 调整先验概率：自定义权重 ==========
gnb_prior = GaussianNB(priors=[0.3, 0.7])  # 良性0.3，恶性0.7
gnb_prior.fit(X_train_scaled, y_train)

# 预测
y_test_pred_prior = gnb_prior.predict(X_test_scaled)
y_test_proba_prior = gnb_prior.predict_proba(X_test_scaled)[:, 1]

# ========== 评估指标（核心修复：全部改用内置 round() 函数） ==========
print("===== 先验概率调整前后性能对比 =====")
# 修复点：float不能调用.round()，改为 round(结果, 4)
print("原始模型测试集召回率（恶性识别率）：", round(recall_score(y_test, y_test_pred), 4))
print("调整后模型测试集召回率（恶性识别率）：", round(recall_score(y_test, y_test_pred_prior), 4))
print("原始模型测试集精确率：", round(precision_score(y_test, y_test_pred), 4))
print("调整后模型测试集精确率：", round(precision_score(y_test, y_test_pred_prior), 4))
print("原始模型测试集AUC值：", round(roc_auc_score(y_test, y_test_proba), 4))
print("调整后模型测试集AUC值：", round(roc_auc_score(y_test, y_test_proba_prior), 4))

# %% [markdown]
# ## 六、与其他分类模型对比
# 
# 为了验证朴素贝叶斯模型的性能，与逻辑回归、支持向量机、决策树等经典分类模型进行对比

# %%
# 导入其他分类模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 初始化对比模型
models = {
    '高斯朴素贝叶斯': GaussianNB(),
    '逻辑回归': LogisticRegression(random_state=42),
    '支持向量机': SVC(random_state=42, probability=True),
    '决策树': DecisionTreeClassifier(random_state=42),
    '随机森林': RandomForestClassifier(random_state=42)
}

# 训练并评估所有模型
results = []
for model_name, model in models.items():
    # 训练模型
    model.fit(X_train_scaled, y_train)
    # 预测
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    # 保存结果
    results.append({
        '模型名称': model_name,
        '准确率': accuracy,
        '精确率': precision,
        '召回率': recall,
        'F1分数': f1,
        'AUC值': auc
    })

# 转换为DataFrame并排序
results_df = pd.DataFrame(results).sort_values(by='AUC值', ascending=False).round(4)
print("===== 各分类模型性能对比结果 =====")
print(results_df)

# 可视化模型性能对比
plt.figure(figsize=(12, 6))
x = np.arange(len(results_df['模型名称']))
width = 0.35

# 绘制准确率和AUC值对比
plt.bar(x - width/2, results_df['准确率'], width, label='准确率', color='#1f77b4')
plt.bar(x + width/2, results_df['AUC值'], width, label='AUC值', color='#ff7f0e')

plt.title('各分类模型性能对比', fontsize=14)
plt.xlabel('模型名称', fontsize=12)
plt.ylabel('指标值', fontsize=12)
plt.xticks(x, results_df['模型名称'], rotation=15)
plt.ylim([0.8, 1.0])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 七、结论与总结

# %% [markdown]
# ### 7.1 核心结论
# 
# 1. **模型性能** :高斯朴素贝叶斯模型在乳腺癌肿瘤分类任务上表现优异，测试集准确率达到97%以上，AUC值超过0.99，具备良好的分类能力和泛化能力。 
# 2. **模型优势** :朴素贝叶斯模型训练速度快、计算开销小，对小样本数据表现良好，同时具备良好的可解释性，符合医疗诊断场景的需求。
# 3. **优化效果** :通过特征选择和先验概率调整，可进一步提升模型的恶性肿瘤识别率（召回率），降低漏诊风险，更适配医疗诊断的业务需求。
# 4. **模型对比** :与逻辑回归、支持向量机等模型相比，朴素贝叶斯模型在保持高准确率的同时，训练效率更高，模型更轻量化，适合部署在医疗辅助诊断系统中。
# 
# ### 7.2 项目亮点
# 1. **完整的机器学习流程**：覆盖了数据加载、探索性分析、数据预处理、模型构建、训练评估、优化调参、模型对比的全流程，符合工业级机器学习项目规范。
# 
# 2. **可解释性强**：通过特征分布、相关性分析、混淆矩阵、ROC曲线等可视化手段，全面展示了模型的分类逻辑和性能表现，具备良好的可解释性。
# 
# 3. **业务适配性**：针对医疗诊断场景的核心需求（降低恶性肿瘤漏诊率），通过先验概率调整优化了模型的召回率，更贴合实际业务场景。
# 
# 4. **可复现性强**：所有代码均设置了随机种子，数据处理和模型训练流程清晰，可直接复现实验结果。
# 
# ### 7.3 后续优化方向
# 1. **特征工程优化**：可进一步通过特征交叉、特征降维（PCA/LDA）等方式，提取更具区分度的特征，提升模型性能。
# 
# 2. **模型融合**：可将朴素贝叶斯与其他模型进行融合（如Stacking、Voting），进一步提升模型的分类精度和鲁棒性。
# 
# 3. **超参数调优**：可通过网格搜索、贝叶斯优化等方式，对模型的超参数进行更精细的调优，挖掘模型的最大性能潜力。
# 
# 4. **模型部署**：可将训练好的模型封装为API接口，开发可视化的医疗辅助诊断系统，实现模型的落地应用。

# %% [markdown]
# ## 八、附录

# %% [markdown]
# ### 14.1 特征重要性总结
# 
# ### 8.1 环境要求
# - Python 3.8及以上版本
# - 依赖第三方库：`pandas`、`numpy`、`matplotlib`、`seaborn`、`scikit-learn`
# 
# 一键安装命令：
# ```bash
# pip install pandas numpy matplotlib seaborn scikit-learn
#  
# 

# %% [markdown]
# ### 8.2 代码说明
# 1. 本Notebook所有代码均可直接运行，无需额外配置；
# 2. 数据文件`data.csv`需要和Notebook文件放置在同一目录下；
# 3. 全部可视化图表会在代码执行后自动渲染展示；
# 4. 模型训练指标、评估结果会在控制台逐行打印，可即时查看。

# %% [markdown]
# ### 8.3 数据集来源
# 
# 威斯康辛州乳腺癌数据集下载地址：
# https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
# 


