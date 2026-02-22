# 基于 RAG 的轻量级多模态文档问答系统

## 项目简介：

针对科研论文、技术报告等 PDF 文档中**文本与图片信息混合存在**、传统问答系统难以覆盖图文内容，以及大模型易产生“幻觉”问题，设计并实现了一套基于检索增强生成（RAG, Retrieval-Augmented Generation）架构的**轻量多模态智能文档问答系统**。

系统通过：

> 文本解析 + 图片 OCR + 语义检索 + 大模型生成 + 溯源机制

实现对**多篇 PDF 文档（含图像内容）**的高精度问答与多轮对话交互，提升知识获取效率与回答可信度。

## 项目流程：

![](E:\wendang\Markdown\图片\英语\workflow.jpg)



```
PDF上传
   ↓
文本解析（PyPDFLoader）
   ↓
图片抽取（PyMuPDF）
   ↓
图像OCR识别（EasyOCR）
   ↓
文本切分（RecursiveCharacterTextSplitter）
   ↓
向量化（MiniLM Embedding）
   ↓
向量存储（FAISS）
   ↓
语义检索（Top-K）
   ↓
大模型生成（DeepSeek API）
   ↓
答案溯源输出
```

## 核心特点

###  1. 多模态支持

- 支持 PDF 文本解析
- 支持 PDF 图片抽取
- 支持图片 OCR 识别
- 图文统一向量化

------

### 2. RAG 架构增强

- 基于向量检索增强生成
- 有效降低大模型幻觉
- 支持 Top-K 语义检索

------

### 3. 答案溯源机制

- 输出引用文档来源
- 显示页码信息
- 提高结果可信度

------

### 4. 多轮对话支持

- 使用 Session State 维护历史
- 支持上下文连续问答



## 所用技术：

**编程语言：** Python 3.10+

**框架：** LangChain 1.x

**向量数据库：** FAISS

**OCR：** EasyOCR

**PDF解析：** PyMuPDF + PyPDFLoader

**向量嵌入：** sentence-transformers/all-MiniLM-L6-v2

**对话模型：** DeepSeek API（deepseek-chat）

**文本切分：** RecursiveCharacterTextSplitter

**Prompt 构建：** ChatPromptTemplate

**输出解析：** StrOutputParser

**前端界面：** Streamlit

**多轮对话管理：** Streamlit Session State

## Requirements

1. **安装依赖**:

   ```bash
   pip install -r requirements.txt
   ```

2. **运行**:

   ```bash
   streamlit run app.py
   ```

3. **使用**：

   .streamlit下面的secrets.toml粘贴你的deep seek api key

   ```toml
   DEEPSEEK_API_KEY="your api key"
   ```

   