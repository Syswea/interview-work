import pandas as pd
import numpy as np
import re
from typing import TypedDict, List, Optional
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 初始化
# ==========================================
llm = ChatOpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="not-needed",
    model_name="local-model",
    temperature=0.1
)

# 读取数据 (请确保 train.csv 在目录下)
df_train_raw = pd.read_csv('./P3/train.csv')

class AgentState(TypedDict):
    code: str
    current_score: float
    feedback: str
    iteration_count: int
    is_fix_needed: bool  # 新增：标记是否需要纠错
    best_code: str

# ==========================================
# 2. 核心函数
# ==========================================

def map3_score(y_true, y_probs, labels):
    top3_idx = np.argsort(y_probs, axis=1)[:, -3:][:, ::-1]
    score = 0.0
    y_true_vals = y_true.values if hasattr(y_true, 'values') else y_true
    for i, true_val in enumerate(y_true_vals):
        prediction_list = labels[top3_idx[i]]
        for j, pred in enumerate(prediction_list):
            if pred == true_val:
                score += 1.0 / (j + 1)
                break
    return score / len(y_true)

def feature_engineer_node(state: AgentState):
    # 根据是否报错调整 Prompt
    error_prefix = ""
    if state.get("is_fix_needed", False):
        error_prefix = f"【紧急纠错】你上一轮的代码报错了：{state['feedback']}\n请修正错误！"
    
    prompt = f"""你是一个 Kaggle 专家。任务：编写 Python 函数 `transform_data(df)`。
原始字段：Temparature, Humidity, Moisture, Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous
{error_prefix}
当前最高分：{state['current_score']}
要求：
1. 包含 import pandas as pd 和 import numpy as np。
2. 必须处理原始特征，返回包含 3 个以上新特征的 DataFrame。
3. 只输出代码，不要任何解释，不要包含 Markdown 标签。
"""
    response = llm.invoke(prompt)
    clean_code = re.sub(r'```python|```', '', response.content).strip()
    return {"code": clean_code}

def evaluation_node(state: AgentState):
    global df_train_raw
    try:
        # 执行 LLM 生成的代码
        exec_globals = {"pd": pd, "np": np}
        exec(state['code'], exec_globals)
        transform_fn = exec_globals['transform_data']
        
        # 数据处理
        df = transform_fn(df_train_raw.copy())
        
        # 简单处理分类变量
        for col in df.select_dtypes(include=['object']).columns:
            if col != 'Fertilizer Name':
                df[col] = df[col].astype('category').cat.codes
        
        X = df.drop(['id', 'Fertilizer Name'], axis=1, errors='ignore')
        y = df['Fertilizer Name']
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LGBMClassifier(n_estimators=100, learning_rate=0.1, verbose=-1)
        model.fit(X_train, y_train)
        
        score = map3_score(y_val, model.predict_proba(X_val), model.classes_)
        
        # 成功运行：不计入纠错，增加迭代计数
        new_best_code = state['best_code']
        if score > state['current_score']:
            new_best_code = state['code']
            
        return {
            "current_score": max(score, state['current_score']),
            "feedback": f"运行成功，得分: {score:.4f}",
            "is_fix_needed": False,
            "iteration_count": state['iteration_count'] + 1,
            "best_code": new_best_code
        }
    except Exception as e:
        # 运行失败：标记需要纠错，不增加 iteration_count
        return {
            "feedback": f"❌ 运行报错: {str(e)}",
            "is_fix_needed": True
        }

# ==========================================
# 3. 路由与流程构建
# ==========================================

def should_continue(state: AgentState):
    # 1. 优先处理纠错
    if state.get("is_fix_needed", False):
        print(f"   >>> 发现错误，打回修正...")
        return "engineer"
    # 2. 判断是否完成
    if state["iteration_count"] >= 5:
        return END
    # 3. 继续新一轮探索
    return "engineer"

workflow = StateGraph(AgentState)
workflow.add_node("engineer", feature_engineer_node)
workflow.add_node("evaluate", evaluation_node)

workflow.set_entry_point("engineer")
workflow.add_edge("engineer", "evaluate")
# 只添加一次路由
workflow.add_conditional_edges("evaluate", should_continue, {
    "engineer": "engineer",
    END: END
})

app = workflow.compile()

# ==========================================
# 4. 执行
# ==========================================

initial_state = {
    "code": "",
    "current_score": 0.0,
    "feedback": "开始探索",
    "iteration_count": 0,
    "is_fix_needed": False,
    "best_code": ""
}

print("🚀 Agent 开始工作，正在实时流式输出节点状态...\n")

# 使用 stream 模式查看每一个步骤
for output in app.stream(initial_state):
    # output 的格式是 { "节点名称": { "状态更新内容" } }
    for node_name, state_update in output.items():
        print(f"标注节点: [{node_name}]")
        
        if node_name == "engineer":
            print("📝 LLM 生成的代码片段 (前 100 字符):")
            print(state_update['code'][:100] + "...")
            
        elif node_name == "evaluate":
            print(f"📊 评估结果: {state_update.get('feedback', '无反馈')}")
            print(f"🏆 当前最佳分数: {state_update.get('current_score', 0.0)}")
            
        print("-" * 40)

# 最后打印最终结果
final_state = app.get_state(config={}).values # 获取最后状态（取决于具体版本，可用 invoke 的结果代替）