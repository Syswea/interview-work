import numpy as np
import pandas as pd

# define files path
train_path = "./train.csv"
test_path = "./test.csv"

# read csv file
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# remove id
train_df.drop(columns=['id'], inplace=True)
test_ids = test_df['id']
test_df.drop(columns=['id'], inplace=True)

print(f"shape of train: {train_df.shape}")
print(f"shape of test: {test_df.shape}")
print(f"columns names: {train_df.columns[:].tolist()}")

def apply_sr_features(df):
    # 避免除以零
    eps = 1e-6
    
    # 1. 养分比例因子 (经典的农学指标)
    df['N_P_ratio'] = df['Nitrogen'] / (df['Phosphorous'] + eps)
    df['N_K_ratio'] = df['Nitrogen'] / (df['Potassium'] + eps)
    df['P_K_ratio'] = df['Phosphorous'] / (df['Potassium'] + eps)
    
    # 2. 环境交互因子
    df['Temp_Hum_Index'] = df['Temparature'] * df['Humidity']
    df['Moisture_Hum_Ratio'] = df['Moisture'] / (df['Humidity'] + eps)
    
    # 3. 养分总量
    df['Total_NPK'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
    
    # 4. 养分集中度 (SR 常用非线性组合)
    df['N_interaction'] = df['Nitrogen'] * df['Moisture']

    # [agent]代码
    
    
    return df

# 对训练集和测试集同时应用
train_df = apply_sr_features(train_df)
test_df = apply_sr_features(test_df)

# 获取所有唯一的肥料名称
unique_fertilizers = train_df['Fertilizer Name'].unique()

print(f"共有 {len(unique_fertilizers)} 种肥料：")
print(unique_fertilizers)

# target variable y
if 'Fertilizer Name' in train_df.columns:
    
    y = train_df['Fertilizer Name'].map({'28-28':0, '17-17-17':1, '10-26-26':2, 'DAP':3, '20-20':4, '14-35-14':5, 'Urea':6})
    train_df.drop(columns=['Fertilizer Name'], inplace=True)

# label encoding
from sklearn.preprocessing import LabelEncoder

cat_cols = train_df.select_dtypes(include=['object']).columns

for col in cat_cols:
    # fill N/A
    train_df[col] = train_df[col].astype(str).fillna('missing')
    test_df[col] = test_df[col].astype(str).fillna('missing')
    
    le = LabelEncoder()

    # get all labels
    full_data = pd.concat([train_df[col], test_df[col]], axis=0)
    le.fit(full_data)
    
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

print("Label Encoding finish")

# split train and val
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    train_df, y, test_size=0.2, random_state=42, stratify=y
)

print(f"shape of X_train: {X_train.shape}")
print(f"shape of X_val: {X_val.shape}")

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
    # --- [GA 部分: 特征选择开关] ---
    all_features = X_train.columns.tolist()
    # 为每一个特征创建一个布尔开关
    selected_features = [
        col for col in all_features 
        if trial.suggest_categorical(f'use_{col}', [True, False])
    ]
    
    # 确保至少选择了一个特征，否则该 trial 无效
    if len(selected_features) == 0:
        return 0

    # 使用选中的特征子集
    X_tr_sub = X_train[selected_features]
    X_va_sub = X_val[selected_features]

    param = {
        'objective': 'multiclass',
        'num_class': 7,
        'metric': 'multi_logloss',
        'verbosity': -1,
        'device': 'gpu', # 小数据集建议用 cpu，减少数据搬运时间
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 64),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'n_estimators': 300, # 配合早停可以使用更大的值
        'random_state': 42
    }

    model = lgb.LGBMClassifier(**param)
    
    # 关键修改：使用 Numpy 数组训练，并加入 eval_set 开启早停提速
    model.fit(
        X_tr_sub, y_train,
        eval_set=[(X_va_sub, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    probs = model.predict_proba(X_va_sub)
    score = mapk(y_val, probs, k=3)
    
    return score

# 6. 开始优化
study_lgb = optuna.create_study(direction='maximize')
study_lgb.optimize(objective_lgb, n_trials=30)

print("最优参数: ", study_lgb.best_params)
print("最高 MAP@3: ", study_lgb.best_value)

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

# --- 1. 从 Optuna 结果中提取被选中的特征名 ---
best_params_all = study_lgb.best_params

# 找出所有值为 True 的 'use_xxx' 参数，并还原成原始列名
selected_features = [
    k.replace('use_', '') for k, v in best_params_all.items() 
    if k.startswith('use_') and v is True
]

print(f"✅ 最终模型选中的特征 ({len(selected_features)}个):")
print(selected_features)

# --- 2. 提取真正的模型超参数 (剔除 use_ 开头的开关) ---
final_model_params = {
    k: v for k, v in best_params_all.items() 
    if not k.startswith('use_')
}
final_model_params.update({
    'objective': 'multiclass',
    'num_class': 7,
    'metric': 'multi_logloss',
    'verbosity': -1,
    'device': 'cpu'  # 最终预测阶段 CPU 足够快且稳
})

# --- 3. 准备全量训练数据 (只包含选中的特征) ---
# 假设 X 是包含所有 SR 特征的完整训练 DataFrame
X_full = pd.concat([X_train, X_val])[selected_features]
y_full = pd.concat([y_train, y_val])

# --- 4. 最终全量训练 ---
print("正在进行最终全量训练...")
final_model = lgb.LGBMClassifier(**final_model_params)
final_model.fit(X_full, y_full)

# --- 5. 对测试集进行预测 (关键：只取选中的特征) ---
print("正在生成预测结果...")
# 这里的 test_df 必须也要用 selected_features 过滤
test_df_filtered = test_df[selected_features]
probs = final_model.predict_proba(test_df_filtered)

# --- 6. 生成提交文件 ---
top3_idx = np.argsort(-probs, axis=1)[:, :3]
inv_map = {0:'28-28', 1:'17-17-17', 2:'10-26-26', 3:'DAP', 4:'20-20', 5:'14-35-14', 6:'Urea'}

final_preds = [" ".join([inv_map[idx] for idx in row]) for row in top3_idx]

submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': final_preds
})

submission.to_csv('submission.csv', index=False)
print("🚀 提交文件 submission.csv 已成功生成！")
display(submission.head())