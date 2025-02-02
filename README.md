# crawl4AI-agent

## 项目介绍
crawl4AI-agent 是一个基于Python的网络爬虫代理，旨在从网站中提取结构化数据并生成嵌入向量。项目结合了ollama的本地LLM和嵌入模型，使用Supabase作为数据库存储，并通过uv进行依赖管理。

## 项目特点
- 使用ollama本地模型进行数据处理和嵌入生成
- 支持从sitemap或单个页面抓取数据
- 集成Supabase进行数据存储
- 使用uv进行高效的依赖管理
- 提供Web UI界面

## 使用方法

### 1. 环境准备
1. 安装 [ollama](https://ollama.ai/) 并拉取所需模型：
   ```bash
   ollama pull llama2
   ollama pull nomic-embed-text
   ```

2. 使用uv创建虚拟环境并同步依赖：
   ```bash
   uv venv
   uv sync
   ```

3. 或者你可以使用conda创建虚拟环境并安装依赖：
   ```bash
   conda create -n crawl4AI python=3.12
   conda activate crawl4AI
   pip install -r requirements.txt
   ```

### 2. Supabase配置
1. 登录Supabase控制台
2. 进入 "SQL Editor" 选项卡
3. 将 `site_pages.sql` 中的代码粘贴到编辑器
4. 点击 "Run" 执行SQL语句创建数据库表

### 3. 运行项目
1. 配置环境变量：
   ```bash
   cp .env.example .env
   # 编辑.env文件填写Supabase和ollama配置
   ```

2. 启动Web UI：
   ```bash
   python webui.py
   ```

3. 运行爬虫示例：
   ```bash
   python examples/crawl_docs_sitemap.py
   ```

## 项目结构
```
.
├── .env                    # 环境变量配置
├── .gitignore
├── .python-version
├── crawl4ai_docs.py        # 主爬虫模块
├── pyproject.toml          # 项目配置
├── rag_agent.py            # RAG代理实现
├── README.md               # 项目文档
├── requirements.txt        # 依赖列表
├── site_pages.sql          # Supabase表结构
├── uv.lock                 # uv锁定文件
├── webui.py                # Web界面
├── examples/               # 示例代码
│   ├── crawl_docs_sitemap.py
│   └── single_page.py
└── test/                   # 测试代码
    ├── test_agent.py
    ├── test_ollama_embed.py
    ├── test_ollama_in_pydanticai.py
    └── test_ollama_json.py
```

## 项目配置

### 数据库架构
数据库表结构定义在 `site_pages.sql` 中，主要包含以下表：
- `site_pages`: 存储抓取的页面信息
- `page_chunks`: 存储分块后的文本内容
- `embeddings`: 存储文本嵌入向量

### Chunking配置
chunking配置可在 `crawl4ai_docs.py` 中调整：
- `CHUNK_SIZE`: 文本分块大小（默认：1000字符）
- `CHUNK_OVERLAP`: 分块重叠大小（默认：200字符）