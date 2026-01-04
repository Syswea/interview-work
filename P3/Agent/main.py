import pandas as pd
import requests
import json

def run_client():
    # 1. 读取当前目录下的 X_train.csv
    file_path = "./X_train.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误：在当前目录下找不到 {file_path}")
        return

    # 2. 将 DataFrame 转换为 JSON 格式
    # 注意：75万行数据很大，转换和传输可能较慢
    data_json = df.to_dict(orient='records')

    # 3. 定义 API 地址
    api_url = "http://127.0.0.1:8000/calculate_map3"

    print("正在发送数据到 API 进行计算，请稍候...")
    
    # 4. 调用 API
    try:
        response = requests.post(api_url, json=data_json, timeout=7200) # 设置较长超时时间
        if response.status_code == 200:
            # 5. 输出 API 返回的结果
            print(response.json()["result"])
        else:
            print(f"请求失败，状态码：{response.status_code}, 详情：{response.text}")
    except Exception as e:
        print(f"发生错误：{e}")

if __name__ == "__main__":
    run_client()