# **项目方案：FinSight —— 智能金融研报生成与风险分析系统**

## **1\. 项目概述 (Executive Summary)**

**FinSight** 是一个面向金融领域的垂直大模型应用。该系统旨在模拟资深金融分析师的工作流程，通过混合架构（Hybrid Architecture）处理金融文本。

系统首先利用传统机器学习模型（**XGBoost**）对输入的财经新闻或财报摘要进行快速风险定性（Sentiment/Risk Analysis），随后利用 **LangGraph** 编排工作流，根据风险等级调用 **RAG**（检索增强生成）获取历史数据与法规背景，最后通过经过 **LoRA** 微调的垂直领域大模型生成专业、合规的分析简报。

### **核心技术亮点**

* **RAG (Retrieval-Augmented Generation)**: 解决金融数据时效性和幻觉问题。  
* **LoRA (Low-Rank Adaptation)**: 低成本赋予模型“金融专家”的语气和术语理解能力。  
* **XGBoost/LightGBM**: 作为“路由大脑”，体现传统 ML 与 LLM 的高效结合。  
* **LangGraph**: 实现复杂的条件分支逻辑（Agentic Workflow）。

## **2\. 系统架构 (System Architecture)**

graph TD  
    User\[用户输入: 财经新闻/财报片段\] \--\> Feature\[特征工程: TF-IDF\]  
    Feature \--\> ML\_Router{XGBoost 分类器}  
      
    ML\_Router \-- "高风险 (Negative)" \--\> Branch\_A\[深度分析模式\]  
    ML\_Router \-- "一般/利好 (Neutral/Positive)" \--\> Branch\_B\[快速摘要模式\]  
      
    subgraph Deep\_Analysis\_Workflow \[LangGraph 工作流\]  
        Branch\_A \--\> RAG\[RAG 检索: 历史违约/法规库\]  
        RAG \--\> Context\[组装上下文\]  
        Context \--\> LoRA\_Model\[LoRA 微调模型\]  
    end  
      
    subgraph Quick\_Summary\_Workflow  
        Branch\_B \--\> LoRA\_Model  
    end  
      
    LoRA\_Model \--\> Final\_Output\[输出: 专业金融简报\]

## **3\. 详细实施方案 (Implementation Plan)**

### **第一阶段：数据获取与工程化 (Data Engineering)**

**目标**：构建用于 ML 训练的结构化数据和用于 RAG 的非结构化知识库。

* **数据源 (Kaggle)**:  
  1. **分类数据**: [Financial PhraseBank](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news) (用于训练 XGBoost)。  
  2. **RAG/LoRA 语料**: [SEC EDGAR Filings](https://www.google.com/search?q=https://www.kaggle.com/datasets/finamys/sec-edgar-filings-2000-2023-text-data) 或金融研报 PDF。  
* **技术栈**: Pandas, NLTK/SpaCy, Python Request  
* **关键动作**:  
  * 清洗 PhraseBank 数据，将标签映射为数字 (0: Negative, 1: Neutral, 2: Positive)。  
  * 编写脚本清洗 SEC 文本，去除 HTML 标签，提取 Item 1 和 Item 7 核心章节。  
* **预期产出**: sentiment\_train.csv 和 清洗后的 corp\_docs/ 文件夹。  
* **预估时间**: 3-5 天

### **第二阶段：构建“路由大脑” (Machine Learning)**

**目标**：训练一个高效率的分类器，用于工作流的前置路由。

* **技术栈**: Scikit-learn (TF-IDF), XGBoost (or LightGBM), Joblib  
* **核心逻辑**:  
  * 不使用 LLM 做简单分类（成本高、速度慢）。  
  * 使用 TF-IDF 将文本向量化，输入 XGBoost 进行三分类预测。  
* **需要理解的知识**:  
  * TF-IDF 原理 vs Word2Vec/Embedding。  
  * 分类模型评价指标 (F1-Score, Confusion Matrix)。  
* **预期产出**: xgboost\_financial\_router.pkl (模型文件)。  
* **预估时间**: 3 天

### **第三阶段：注入“专家灵魂” (LoRA Fine-tuning)**

**目标**：微调基座模型，使其掌握金融术语、黑话及特定回复格式。

* **技术栈**: Unsloth (推荐) 或 LLaMA-Factory, HuggingFace Transformers  
* **基座模型**: Llama-3-8B-Instruct 或 Qwen-2.5-7B  
* **关键动作**:  
  1. **构建指令集 (Instruction Dataset)**: 制作 JSONL 格式数据。  
     * *Input*: "分析特斯拉 Q3 财报..."  
     * *Output*: (模仿高盛/大摩分析师的语气，使用 EBITDA、EPS 等术语)。  
  2. **LoRA 训练**: 设置秩 r=16, alpha=32，在 Colab (T4 GPU) 或本地显卡上训练。  
* **需要理解的知识**:  
  * LoRA/QLoRA 的数学原理（低秩矩阵分解）。  
  * Prompt Engineering (System Prompt 设计)。  
* **预期产出**: lora\_adapter/ 权重文件夹。  
* **预估时间**: 7 天

### **第四阶段：构建“知识外挂” (RAG System)**

**目标**：建立向量检索系统，提供外部知识支持。

* **技术栈**: LangChain, ChromaDB (向量库), BGE-M3 (Embedding 模型)  
* **关键动作**:  
  1. **Chunking**: 使用 RecursiveCharacterTextSplitter 将长文档切分为 500-1000 token 的块。  
  2. **Embedding**: 向量化并存入 ChromaDB。  
  3. **Retrieval**: 实现 get\_relevant\_documents 函数。  
* **需要理解的知识**:  
  * 向量相似度计算 (Cosine Similarity)。  
  * Chunk overlap (块重叠) 的意义。  
* **预期产出**: 本地向量数据库文件。  
* **预估时间**: 3-5 天

### **第五阶段：核心编排 (LangGraph Integration)**

**目标**：将 ML、RAG、LoRA 串联为一个有状态的智能体。

* **技术栈**: LangGraph, LangChain Core  
* **节点设计 (Nodes)**:  
  1. classifier\_node: 调用 XGBoost 模型，更新 State 中的 risk\_level。  
  2. retrieve\_node: (条件触发) 查询 ChromaDB。  
  3. generate\_node: 加载 LoRA 模型，结合 Context 生成最终回复。  
* **边设计 (Edges)**:  
  * 基于 risk\_level 的条件边 (Conditional Edge)。  
* **预期产出**: 完整的 main.py 应用脚本。  
* **预估时间**: 5-7 天

## **4\. 技术栈清单 (Tech Stack Checklist)**

| 类别 | 工具/库 | 用途 |
| :---- | :---- | :---- |
| **语言** | Python 3.10+ | 核心开发语言 |
| **数据处理** | Pandas, NumPy | 数据清洗与结构化处理 |
| **传统 ML** | Scikit-learn, XGBoost | 文本特征提取与意图/风险分类 |
| **LLM 微调** | Unsloth, PEFT, PyTorch | 高效 LoRA 微调框架 |
| **LLM 编排** | LangChain, LangGraph | 构建有环、有状态的工作流 |
| **向量数据库** | ChromaDB / FAISS | RAG 知识库存储 |
| **基座模型** | Llama-3 / Qwen-2.5 | 开源大模型基座 |

## **5\. 项目简历/面试价值 (Resume Value)**

完成此项目后，您可以在简历中展示以下关键能力：

1. **Full-Stack AI Engineering**: 不仅会调 API，还能从数据清洗到模型微调全流程落地。  
2. **Cost-Efficiency Awareness**: 懂得使用 XGBoost 做前置过滤，而不是盲目使用大模型，体现了工程化思维（降低 Token 消耗和延迟）。  
3. **Vertical Domain Adaptation**: 展示了如何通过 LoRA 将通用大模型落地到特定垂直领域（金融）。  
4. **Complex Logic Handling**: 通过 LangGraph 证明了处理非线性、复杂业务逻辑的能力。

### **下一步建议**

建议您按照 **阶段 2 (XGBoost) \-\> 阶段 4 (RAG) \-\> 阶段 3 (LoRA) \-\> 阶段 5 (整合)** 的顺序进行开发。这样可以先从容易上手的模块建立信心，最后再攻克微调和编排的难点。