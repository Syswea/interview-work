import numpy as np
import pandas as pd

df = pd.DataFrame()

df.drop(columns=['id'], inplace=True)


# target variable y
if 'class' in df.columns:
    
    y = df['class']
    df.drop(columns=['class'], inplace=True)

# split train and val
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    df, y, test_size=0.2, random_state=42, stratify=y
)

import optuna
import lightgbm as lgb
import numpy as np

# 1. 定义 MAP@3 计算函数
def mapk(actual, predicted_probs, k=3):
    """
    actual: 真实标签的数组 (n_samples,)
    predicted_probs: 模型输出的概率矩阵 (n_samples, n_classes)
    """
    scores = []
    # 获取概率最高的前 k 个索引
    top_k_indices = np.argsort(-predicted_probs, axis=1)[:, :k]
    
    for a, p in zip(actual, top_k_indices):
        score = 0.0
        for i, pred in enumerate(p):
            if pred == a:
                score = 1.0 / (i + 1)
                break
        scores.append(score)
    return np.mean(scores)

def objective_lgb(trial):
    # 2. 修改参数搜索空间为多分类
    param = {
        'objective': 'multiclass',
        'num_class': 7,  # 你的肥料种类总数
        'metric': 'multi_logloss',
        'verbosity': -1,
        'device': 'gpu', # 如果没有 GPU 请改为 cpu
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': 200,
        'random_state': 42
    }

    # 如果数据集很大，也可以只用一组简单的 train_test_split
    model = lgb.LGBMClassifier(**param)
    
    # 假设使用 X_train, y_train 进行简单的验证
    model.fit(X_train, y_train)
    
    # 4. 获取概率矩阵
    probs = model.predict_proba(X_val)
    
    # 5. 计算 MAP@3
    score = mapk(y_val, probs, k=3)
    
    return score

# 6. 开始优化
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)

print("最高 MAP@3: ", study_lgb.best_value)