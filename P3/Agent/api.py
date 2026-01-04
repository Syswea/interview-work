import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, Body
from typing import List, Dict

app = FastAPI()

@app.post("/calculate_map3")
async def calculate_map3(data: List[Dict] = Body(...)):
    # 1. 接收数据并转换为 DataFrame
    df = pd.DataFrame(data)

    # --- 原始逻辑开始 ---
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    if 'class' in df.columns:
        y = df['class']
        df.drop(columns=['class'], inplace=True)

    X_train_split, X_val, y_train_split, y_val = train_test_split(
        df, y, test_size=0.2, random_state=42, stratify=y
    )

    def mapk(actual, predicted_probs, k=3):
        scores = []
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
        param = {
            'objective': 'multiclass',
            'num_class': 7,
            'metric': 'multi_logloss',
            'verbosity': -1,
            'device': 'gpu', 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 64),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'n_estimators': 100,
            'random_state': 42
        }
        model = lgb.LGBMClassifier(**param)
        model.fit(X_train_split, y_train_split)
        probs = model.predict_proba(X_val)
        return mapk(y_val, probs, k=3)

    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lgb, n_trials=3)
    # --- 原始逻辑结束 ---

    # 按要求输出
    output = f"最高 MAP@3: {study_lgb.best_value}"
    return {"result": output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)