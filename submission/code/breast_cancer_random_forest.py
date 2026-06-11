# ╔══════════════════════════════════════════════════════════════╗
# ║  自动安装依赖库 —— 首次运行请执行此单元格                     ║
# ║  如果已安装可跳过 (Ctrl+Enter 运行，无报错即可进入下一步)     ║
# ╚══════════════════════════════════════════════════════════════╝

import sys
import subprocess

required = {
    'numpy':        'numpy',
    'pandas':       'pandas',
    'matplotlib':   'matplotlib',
    'seaborn':      'seaborn',
    'scikit-learn': 'sklearn',
    'xgboost':      'xgboost',
}

missing = []
for pkg_name, import_name in required.items():
    try:
        __import__(import_name)
        print(f'  [OK] {pkg_name} 已安装')
    except ImportError:
        print(f'  [MISSING] {pkg_name}')
        missing.append(pkg_name)

if missing:
    print(f'\n正在安装 {len(missing)} 个缺失的库 ({", ".join(missing)})...')
    # 依次尝试多个源，直到成功
    mirrors = [
        [],                                                          # 默认 PyPI
        ['-i', 'https://pypi.tuna.tsinghua.edu.cn/simple'],          # 清华
        ['-i', 'https://mirrors.ustc.edu.cn/pypi/web/simple'],       # 中科大
    ]
    base_cmd = [sys.executable, '-m', 'pip', 'install',
                '--default-timeout=120', '--retries=3'] + missing

    success = False
    for mirror in mirrors:
        cmd = base_cmd + mirror
        src = mirror[1] if mirror else 'PyPI (默认源)'
        print(f'  尝试: {src}')
        try:
            subprocess.check_call(cmd)
            success = True
            break
        except subprocess.CalledProcessError:
            print(f'  失败，换下一个源...')

    if success:
        print('\n✅ 所有依赖库安装完成！')
        print('请 重启内核 (顶部菜单 Kernel > Restart Kernel) 后跳过此单元格继续运行。')
    else:
        print('\n❌ 自动安装失败。请手动在终端运行:')
        print(f'  {sys.executable} -m pip install {" ".join(missing)}')
else:
    print('\n✅ 所有依赖库已就绪，请继续执行后续单元格！')


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, precision_recall_curve, auc)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os, shutil
import warnings
warnings.filterwarnings('ignore')

# ==================== 中文字体配置（彻底解决方块/白框问题）====================
# 步骤1：动态注册 simhei.ttf 到当前 matplotlib 字体管理器
simhei_path = os.path.join(os.environ['WINDIR'], 'Fonts', 'simhei.ttf')
print(f'[字体] 字体文件路径: {simhei_path}')
print(f'[字体] 是否存在: {os.path.exists(simhei_path)}')

try:
    fm.fontManager.addfont(simhei_path)
    print('[字体] addfont() 成功注册 SimHei')
except Exception as e:
    print(f'[字体] addfont() 失败: {e}，尝试备用方案...')
    try:
        cache_dir = os.path.expanduser('~/.matplotlib')
        os.makedirs(cache_dir, exist_ok=True)
        shutil.copy2(simhei_path, os.path.join(cache_dir, 'simhei.ttf'))
        for fn in os.listdir(cache_dir):
            if fn.startswith('fontlist'):
                os.remove(os.path.join(cache_dir, fn))
        fm._load_fontmanager(try_read_cache=False)
        fm.fontManager.addfont(simhei_path)
        print('[字体] 备用方案注册成功')
    except Exception as e2:
        print(f'[字体] 所有方案均失败: {e2}')

# 步骤2：全局设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 步骤3：sns.set_style 会覆盖 rcParams，所以必须在它之前设置字体并在之后重新确认
sns.set_style('whitegrid')

# 步骤4：重新强制应用字体（sns.set_style 可能覆盖了字体设置）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# 验证
simhei_available = any('simhei' in f.name.lower() for f in fm.fontManager.ttflist)
print(f'[字体] SimHei是否在可用字体列表中: {simhei_available}')
print(f'[字体] 当前 sans-serif 配置: {plt.rcParams["font.sans-serif"][:3]}')

# 其他全局设置
plt.rcParams['figure.dpi'] = 120
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (12, 6)

print('所有依赖库导入成功!')


# 读取 WDBC 数据
column_names = ['id', 'diagnosis',
                'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
                'symmetry_mean', 'fractal_dimension_mean',
                'radius_se', 'texture_se', 'perimeter_se', 'area_se',
                'smoothness_se', 'compactness_se', 'concavity_se', 'concave_points_se',
                'symmetry_se', 'fractal_dimension_se',
                'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
                'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave_points_worst',
                'symmetry_worst', 'fractal_dimension_worst']

df = pd.read_csv('wdbc.data', header=None, names=column_names)
print(f"数据集形状: {df.shape}")
print(f"\n前5行数据:")
df.head()


# 基本统计信息
print("=== 数据类型 ===")
print(df.dtypes.value_counts())
print(f"\n=== 缺失值检查 ===")
print(df.isnull().sum().sum(), "个缺失值")

# 类别分布
print(f"\n=== 类别分布 ===")
print(df['diagnosis'].value_counts())
print(f"\n良性(B)占比: {df['diagnosis'].value_counts(normalize=True)['B']:.2%}")
print(f"恶性(M)占比: {df['diagnosis'].value_counts(normalize=True)['M']:.2%}")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 柱状图
colors_map = {'B': '#2ecc71', 'M': '#e74c3c'}
counts = df['diagnosis'].value_counts()
ax1 = axes[0]
bars = ax1.bar(['良性 (B)', '恶性 (M)'], counts.values, color=[colors_map['B'], colors_map['M']], edgecolor='white', linewidth=1.5)
for bar, count in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{count}\n({count/569:.1%})',
             ha='center', va='bottom', fontsize=13, fontweight='bold')
ax1.set_ylabel('样本数量', fontsize=13)
ax1.set_title('类别分布 (柱状图)', fontsize=14, fontweight='bold')
ax1.set_ylim(0, max(counts.values) * 1.15)

# 饼图
ax2 = axes[1]
wedges, texts, autotexts = ax2.pie(counts.values, labels=['良性 (B)', '恶性 (M)'],
                                    colors=[colors_map['B'], colors_map['M']],
                                    autopct='%1.1f%%', startangle=90,
                                    explode=(0, 0.05), shadow=True)
for autotext in autotexts:
    autotext.set_fontsize(13)
    autotext.set_fontweight('bold')
ax2.set_title('类别分布 (饼图)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('class_distribution.png', bbox_inches='tight', facecolor='white')
plt.show()


# 按诊断结果分组统计
mean_features = [c for c in df.columns if c.endswith('_mean')]
print("=== 良性 vs 恶性: 均值特征对比 ===")
df.groupby('diagnosis')[mean_features].mean().T.style.background_gradient(cmap='RdYlGn_r', axis=1)


# 详细统计
desc = df.drop(columns=['id', 'diagnosis']).describe().T
desc['range'] = desc['max'] - desc['min']
desc[['mean', 'std', 'min', 'max', 'range']].head(10)


# 选取代表性特征进行箱线图对比
key_features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean']

fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    ax = axes[i]
    bp = ax.boxplot([df[df['diagnosis']=='B'][feat], df[df['diagnosis']=='M'][feat]],
                    labels=['良性', '恶性'], patch_artist=True,
                    boxprops=dict(facecolor='#3498db', alpha=0.6),
                    medianprops=dict(color='darkred', linewidth=2))
    # 给第一个箱体绿色
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][0].set_alpha(0.6)
    ax.set_title(feat.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    ax.set_ylabel('特征值')

plt.suptitle('关键均值特征: 良性 vs 恶性', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('boxplot_features.png', bbox_inches='tight', facecolor='white')
plt.show()


fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
for i, feat in enumerate(key_features):
    ax = axes[i]
    parts = ax.violinplot([df[df['diagnosis']=='B'][feat], df[df['diagnosis']=='M'][feat]],
                          positions=[1, 2], showmeans=True, showmedians=True)
    parts['bodies'][0].set_facecolor('#2ecc71')
    parts['bodies'][0].set_alpha(0.7)
    parts['bodies'][1].set_facecolor('#e74c3c')
    parts['bodies'][1].set_alpha(0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(['良性', '恶性'], fontsize=11)
    ax.set_title(feat.replace('_', ' ').title(), fontsize=12, fontweight='bold')

plt.suptitle('关键均值特征: 小提琴图分布', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('violin_features.png', bbox_inches='tight', facecolor='white')
plt.show()


# 计算特征相关性
feature_cols = [c for c in df.columns if c not in ['id', 'diagnosis']]
corr_matrix = df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(250, 15, s=75, l=40, n=16, center='light')
sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, annot=False,
            square=True, linewidths=0.5, cbar_kws={'shrink': 0.8},
            xticklabels=True, yticklabels=True)
ax.set_title('特征相关性热力图', fontsize=18, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', bbox_inches='tight', facecolor='white')
plt.show()


# 找出与 radius_mean 高度相关的特征
high_corr = corr_matrix['radius_mean'].abs().sort_values(ascending=False)
print("与 radius_mean 相关性最高的特征 (Top 10):")
print(high_corr.head(10))
print(f"\n高相关性 (|r| > 0.9) 的特征对数量: {(corr_matrix.abs() > 0.9).sum().sum() / 2:.0f}")


# 标签编码: B -> 0, M -> 1
le = LabelEncoder()
y = le.fit_transform(df['diagnosis'])
print(f"标签映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")
print(f"良性(B->0): {sum(y==0)} 例, 恶性(M->1): {sum(y==1)} 例")

# 特征矩阵
X = df.drop(columns=['id', 'diagnosis']).values

# 标准化 (Z-score)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n标准化完成! 特征均值: {X_scaled.mean():.6f}, 标准差: {X_scaled.std():.6f}")


# 80% 训练, 20% 测试, 保持类别比例
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本 (良性: {sum(y_train==0)}, 恶性: {sum(y_train==1)})")
print(f"测试集: {X_test.shape[0]} 样本 (良性: {sum(y_test==0)}, 恶性: {sum(y_test==1)})")
print(f"训练集良性占比: {sum(y_train==0)/len(y_train):.1%}")
print(f"测试集良性占比: {sum(y_test==0)/len(y_test):.1%}")


# 基础随机森林模型
rf_base = RandomForestClassifier(
    n_estimators=100,    # 决策树数量
    max_depth=None,      # 不限制深度
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
y_prob_base = rf_base.predict_proba(X_test)[:, 1]

print("=== 基础随机森林模型评估 ===")
print(f"训练集准确率: {rf_base.score(X_train, y_train):.4f}")
print(f"测试集准确率: {accuracy_score(y_test, y_pred_base):.4f}")
print(f"精确率 (Precision): {precision_score(y_test, y_pred_base):.4f}")
print(f"召回率 (Recall): {recall_score(y_test, y_pred_base):.4f}")
print(f"F1 分数: {f1_score(y_test, y_pred_base):.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob_base):.4f}")


cm = confusion_matrix(y_test, y_pred_base)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 数值混淆矩阵
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['良性 (预测)', '恶性 (预测)'],
            yticklabels=['良性 (实际)', '恶性 (实际)'],
            cbar=False, annot_kws={'size': 20, 'fontweight': 'bold'})
ax1.set_title('混淆矩阵 (数值)', fontsize=14, fontweight='bold')

# 百分比混淆矩阵
ax2 = axes[1]
cm_pct = cm / cm.sum()
sns.heatmap(cm_pct, annot=True, fmt='.1%', cmap='Blues', ax=ax2,
            xticklabels=['良性 (预测)', '恶性 (预测)'],
            yticklabels=['良性 (实际)', '恶性 (实际)'],
            cbar=False, annot_kws={'size': 16, 'fontweight': 'bold'})
ax2.set_title('混淆矩阵 (百分比)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight', facecolor='white')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC 曲线
ax1 = axes[0]
fpr, tpr, _ = roc_curve(y_test, y_prob_base)
roc_auc = auc(fpr, tpr)
ax1.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'随机森林 (AUC = {roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='随机猜测')
ax1.fill_between(fpr, tpr, alpha=0.2, color='#e74c3c')
ax1.set_xlabel('假阳性率 (FPR)', fontsize=12)
ax1.set_ylabel('真阳性率 (TPR)', fontsize=12)
ax1.set_title('ROC 曲线', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11)
ax1.grid(True, alpha=0.3)

# Precision-Recall 曲线
ax2 = axes[1]
precision, recall, _ = precision_recall_curve(y_test, y_prob_base)
pr_auc = auc(recall, precision)
ax2.plot(recall, precision, color='#3498db', lw=2.5, label=f'随机森林 (AUC = {pr_auc:.4f})')
ax2.axhline(y=sum(y_test)/len(y_test), color='k', ls='--', lw=1.5, alpha=0.5,
            label=f'基准线 ({sum(y_test)/len(y_test):.2f})')
ax2.fill_between(recall, precision, alpha=0.2, color='#3498db')
ax2.set_xlabel('召回率 (Recall)', fontsize=12)
ax2.set_ylabel('精确率 (Precision)', fontsize=12)
ax2.set_title('Precision-Recall 曲线', fontsize=14, fontweight='bold')
ax2.legend(loc='lower left', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('roc_pr_curves.png', bbox_inches='tight', facecolor='white')
plt.show()


print("=== 详细分类报告 ===")
print(classification_report(y_test, y_pred_base, target_names=['良性 (B)', '恶性 (M)']))

# 错误分类样本分析
test_indices = df.iloc[y_test.shape[0]:].index[:len(y_test)] if False else None
misclassified = (y_test != y_pred_base).sum()
print(f"\n错误分类样本数: {misclassified} / {len(y_test)} ({misclassified/len(y_test):.2%})")


# 获取特征重要性
feature_names = [c for c in df.columns if c not in ['id', 'diagnosis']]
importances = rf_base.feature_importances_
indices = np.argsort(importances)[::-1]

# 全部30个特征的重要性
fig, ax = plt.subplots(figsize=(12, 10))
colors = plt.cm.RdYlGn(importances[indices] / importances.max())
ax.barh(range(30), importances[indices], color=colors, edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(30))
ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title('随机森林特征重要性排名 (全部 30 个特征)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

# 在条形图上标注数值
for i, v in enumerate(importances[indices]):
    ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance_full.png', bbox_inches='tight', facecolor='white')
plt.show()


# Top 15 特征重要性
top_n = 15
fig, ax = plt.subplots(figsize=(10, 6))
top_features = [feature_names[i] for i in indices[:top_n]]
top_importances = importances[indices[:top_n]]
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 1, top_n))
bars = ax.barh(range(top_n), top_importances[::-1], color=colors[::-1], edgecolor='gray', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_features[::-1], fontsize=11)
ax.set_xlabel('特征重要性', fontsize=12)
ax.set_title(f'Top {top_n} 最重要特征', fontsize=14, fontweight='bold')
ax.invert_yaxis()

for bar, val in zip(bars, top_importances[::-1]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance_top15.png', bbox_inches='tight', facecolor='white')
plt.show()


# 特征组重要性分析
prefix_groups = {'radius': [], 'texture': [], 'perimeter': [], 'area': [],
                 'smoothness': [], 'compactness': [], 'concavity': [],
                 'concave_points': [], 'symmetry': [], 'fractal_dimension': []}

for name, imp in zip(feature_names, importances):
    for prefix in prefix_groups:
        if name.startswith(prefix):
            prefix_groups[prefix].append(imp)

group_importances = {k: sum(v)/len(v) for k, v in prefix_groups.items() if v}
print("=== 各特征组 (mean + SE + worst) 的平均重要性 ===")
for k, v in sorted(group_importances.items(), key=lambda x: x[1], reverse=True):
    print(f"{k:25s}: {v:.4f}")


# 超参数网格
param_grid = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 先用 RandomizedSearchCV 快速缩小范围
from sklearn.model_selection import RandomizedSearchCV
rf_random = RandomizedSearchCV(
    rf, param_distributions=param_grid, n_iter=30,
    cv=cv, scoring='roc_auc', n_jobs=-1,
    random_state=42, verbose=0
)
rf_random.fit(X_train, y_train)

print("=== RandomizedSearchCV 最佳参数 ===")
for k, v in rf_random.best_params_.items():
    print(f"  {k}: {v}")
print(f"最佳交叉验证 AUC-ROC: {rf_random.best_score_:.4f}")


# 使用最佳参数附近进行精细搜索
best_params = rf_random.best_params_
fine_param_grid = {
    'n_estimators': [max(50, best_params['n_estimators']-50),
                     best_params['n_estimators'],
                     best_params['n_estimators']+50],
    'max_depth': [best_params['max_depth']],
    'min_samples_split': [max(2, best_params['min_samples_split']-1),
                          best_params['min_samples_split']],
    'min_samples_leaf': [max(1, best_params['min_samples_leaf']-1),
                         best_params['min_samples_leaf']],
    'max_features': [best_params['max_features']]
}

grid_search = GridSearchCV(
    rf, fine_param_grid, cv=cv, scoring='roc_auc',
    n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

print("=== 最终最佳参数 ===")
for k, v in grid_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"最佳交叉验证 AUC-ROC: {grid_search.best_score_:.4f}")

rf_best = grid_search.best_estimator_


y_pred_opt = rf_best.predict(X_test)
y_prob_opt = rf_best.predict_proba(X_test)[:, 1]

print("=== 最优随机森林模型评估 ===")
print(f"测试集准确率 (Accuracy):  {accuracy_score(y_test, y_pred_opt):.4f}")
print(f"精确率 (Precision):       {precision_score(y_test, y_pred_opt):.4f}")
print(f"召回率 (Recall):          {recall_score(y_test, y_pred_opt):.4f}")
print(f"F1 分数:                  {f1_score(y_test, y_pred_opt):.4f}")
print(f"AUC-ROC:                  {roc_auc_score(y_test, y_prob_opt):.4f}")

print(f"\n=== 与基础模型对比 ===")
print(f"基础模型准确率: {accuracy_score(y_test, y_pred_base):.4f}  ->  最优模型准确率: {accuracy_score(y_test, y_pred_opt):.4f}")
print(f"基础模型 AUC:    {roc_auc_score(y_test, y_prob_base):.4f}  ->  最优模型 AUC:    {roc_auc_score(y_test, y_prob_opt):.4f}")


# 5 折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_best, X_scaled, y, cv=cv, scoring='accuracy')
cv_auc = cross_val_score(rf_best, X_scaled, y, cv=cv, scoring='roc_auc')

print("=== 5 折交叉验证结果 ===")
print(f"各折准确率: {cv_scores}")
print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print(f"各折 AUC: {cv_auc}")
print(f"平均 AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std() * 2:.4f})")


# 交叉验证可视化
fig, ax = plt.subplots(figsize=(8, 5))
folds = range(1, 6)
ax.plot(folds, cv_scores, 'o-', color='#3498db', linewidth=2.5, markersize=10, label=f'Accuracy (mean={cv_scores.mean():.3f})')
ax.plot(folds, cv_auc, 's-', color='#e74c3c', linewidth=2.5, markersize=10, label=f'AUC (mean={cv_auc.mean():.3f})')
ax.axhline(y=cv_scores.mean(), color='#3498db', linestyle='--', alpha=0.5)
ax.axhline(y=cv_auc.mean(), color='#e74c3c', linestyle='--', alpha=0.5)
ax.set_xlabel('折数', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('5 折交叉验证结果', fontsize=14, fontweight='bold')
ax.set_xticks(folds)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.9, 1.02)
plt.tight_layout()
plt.savefig('cross_validation.png', bbox_inches='tight', facecolor='white')
plt.show()


# 定义多种分类器
classifiers = {
    '随机森林 (Random Forest)': RandomForestClassifier(**grid_search.best_params_, random_state=42, n_jobs=-1),
    '逻辑回归 (Logistic Regression)': LogisticRegression(max_iter=2000, random_state=42),
    'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
    'SVM (Linear Kernel)': SVC(kernel='linear', probability=True, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=10)': KNeighborsClassifier(n_neighbors=10),
}

results = []
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None

    cv_scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy')

    results.append({
        '分类器': name,
        '准确率': accuracy_score(y_test, y_pred),
        '精确率': precision_score(y_test, y_pred),
        '召回率': recall_score(y_test, y_pred),
        'F1分数': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob) if y_prob is not None else 0,
        'CV均值': cv_scores.mean(),
        'CV标准差': cv_scores.std()
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('AUC', ascending=False).reset_index(drop=True)
results_df


# 多模型 ROC 曲线对比
fig, ax = plt.subplots(figsize=(10, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for idx, (name, clf) in enumerate(classifiers.items()):
    if hasattr(clf, 'predict_proba'):
        y_prob = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f'{name} (AUC={roc_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4, label='随机猜测')
ax.set_xlabel('假阳性率 (FPR)', fontsize=12)
ax.set_ylabel('真阳性率 (TPR)', fontsize=12)
ax.set_title('多分类器 ROC 曲线对比', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('multimodel_roc.png', bbox_inches='tight', facecolor='white')
plt.show()


# 分类器性能条形图
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['准确率', '精确率', '召回率', 'F1分数']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics):
    bars = ax.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.85)

ax.set_xlabel('分类器', fontsize=12)
ax.set_ylabel('分数', fontsize=12)
ax.set_title('多分类器性能对比', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(results_df['分类器'], rotation=15, ha='right', fontsize=9)
ax.legend(fontsize=10)
ax.set_ylim(0.85, 1.02)
ax.grid(True, alpha=0.2, axis='y')
plt.tight_layout()
plt.savefig('classifier_comparison.png', bbox_inches='tight', facecolor='white')
plt.show()


# PCA 降维到 2 维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA 前两个主成分的方差解释比例: {pca.explained_variance_ratio_}")
print(f"累计方差解释比例: {pca.explained_variance_ratio_.sum():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# PCA 散点图
ax1 = axes[0]
colors_pca = ['#2ecc71' if label == 0 else '#e74c3c' for label in y]
ax1.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='#2ecc71', alpha=0.6, s=40,
            edgecolors='white', linewidth=0.5, label='良性')
ax1.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='#e74c3c', alpha=0.6, s=40,
            edgecolors='white', linewidth=0.5, label='恶性')
ax1.set_xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax1.set_ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax1.set_title('PCA 降维可视化 (2D)', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11, markerscale=1.5)
ax1.grid(True, alpha=0.3)

# PCA 带决策边界
ax2 = axes[1]
rf_pca = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_pca.fit(X_pca, y)

xx, yy = np.meshgrid(np.linspace(X_pca[:, 0].min()-3, X_pca[:, 0].max()+3, 200),
                     np.linspace(X_pca[:, 1].min()-3, X_pca[:, 1].max()+3, 200))
Z = rf_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

ax2.contourf(xx, yy, Z, alpha=0.3, levels=[-0.5, 0.5, 1.5], colors=['#2ecc71', '#e74c3c'])
ax2.scatter(X_pca[y==0, 0], X_pca[y==0, 1], c='#2ecc71', alpha=0.6, s=40,
            edgecolors='white', linewidth=0.5, label='良性')
ax2.scatter(X_pca[y==1, 0], X_pca[y==1, 1], c='#e74c3c', alpha=0.6, s=40,
            edgecolors='white', linewidth=0.5, label='恶性')
pca_acc = accuracy_score(y, rf_pca.predict(X_pca))
ax2.set_xlabel(f'第一主成分 ({pca.explained_variance_ratio_[0]:.1%})', fontsize=11)
ax2.set_ylabel(f'第二主成分 ({pca.explained_variance_ratio_[1]:.1%})', fontsize=11)
ax2.set_title(f'PCA + 随机森林决策边界 (准确率={pca_acc:.2%})', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11, markerscale=1.5)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_visualization.png', bbox_inches='tight', facecolor='white')
plt.show()


# PCA 各主成分方差解释
fig, ax = plt.subplots(figsize=(10, 5))
pca_full = PCA().fit(X_scaled)
cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

ax.bar(range(1, 31), pca_full.explained_variance_ratio_, alpha=0.7, color='#3498db',
       label='单个主成分方差解释')
ax.step(range(1, 31), cumsum_var, where='mid', color='#e74c3c', lw=2.5,
        label=f'累计方差解释')
ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% 阈值')

# 标注 90% 所需的主成分数
n90 = np.argmax(cumsum_var >= 0.9) + 1
ax.axvline(x=n90, color='gray', linestyle='--', alpha=0.5)
ax.annotate(f'{n90} 个主成分\n解释 90% 方差', xy=(n90, 0.9),
            xytext=(n90+3, 0.85), arrowprops=dict(arrowstyle='->', color='gray'),
            fontsize=11)

ax.set_xlabel('主成分编号', fontsize=12)
ax.set_ylabel('方差解释比例', fontsize=12)
ax.set_title(f'PCA 方差解释分析 (累计 90% 需要 {n90} 个主成分)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.2)
ax.set_xlim(0, 31)
plt.tight_layout()
plt.savefig('pca_variance.png', bbox_inches='tight', facecolor='white')
plt.show()


from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    rf_best, X_scaled, y, cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

fig, ax = plt.subplots(figsize=(10, 6))
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                alpha=0.2, color='#3498db')
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                alpha=0.2, color='#e74c3c')
ax.plot(train_sizes, train_mean, 'o-', color='#3498db', lw=2.5, label='训练集准确率')
ax.plot(train_sizes, test_mean, 's-', color='#e74c3c', lw=2.5, label='交叉验证准确率')

ax.set_xlabel('训练样本数', fontsize=12)
ax.set_ylabel('准确率', fontsize=12)
ax.set_title('学习曲线', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.9, 1.02)
plt.tight_layout()
plt.savefig('learning_curve.png', bbox_inches='tight', facecolor='white')
plt.show()

