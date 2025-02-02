import os
import asyncio
import json
import requests
from xml.etree import ElementTree
from datetime import datetime, timezone
from urllib.parse import urlparse
from dataclasses import dataclass
from typing import List, Dict, Any

from ollama import AsyncClient
from openai import AsyncOpenAI
from supabase import create_client, Client
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

# 初始化Ollama,OpenAI和Supabase客户端
ollama_client = AsyncClient(host=os.getenv('OLLAMA_SERVER_URL'))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]


def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """将文本分割成指定大小的块。"""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # 计算结束位置
        end = start + chunk_size

        # 如果我们已经到达文本的末尾，直接取剩余的部分
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # 首先尝试找到代码块边界（```）
        chunk = text[start:end]
        code_block = chunk.rfind('```')     # rfind方法返回该标志在chunk中的位置（索引），如果未找到则返回-1。
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # 如果没有代码块，尝试在段落边界分割（\n\n）
        elif '\n\n' in chunk:
            # 找到最后一个段落边界
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # 仅在超过chunk_size的30%时才分割
                end = start + last_break

        # 如果没有段落边界，尝试在句子边界分割（. ）
        elif '. ' in chunk:
            # 找到最后一个句子边界
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # 仅在超过chunk_size的30%时才分割
                end = start + last_period + 1

        # 提取块并清理
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # 移动起始位置以获取下一个块
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """使用Ollama提取标题和摘要。"""
    system_prompt = """你是一个AI，用于从文档块中提取标题和摘要。
    返回一个包含'title'和'summary'键的JSON对象。
    对于标题：如果看起来像是文档的开头，请提取其标题。如果是中间块，请推导一个描述性的标题。
    对于摘要：请创建一个简洁的摘要，概括该块的主要内容。
    保持标题和摘要都简洁但有信息量。"""
    
    try:
        response = await ollama_client.chat(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}  
            ],
            format='json'
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        print(f"获取标题和摘要时出错: {e}")
        return {"title": "处理标题出错", "summary": "处理摘要出错"}
    
async def get_embedding(text: str) -> List[float]:
    """从本地Ollama Embedding模型获取文本的嵌入向量。"""
    try:
        response = await ollama_client.embed(
            model="nomic-embed-text:latest",
            input=text
        )
        embedding = response['embeddings'][0]
        return embedding
    except Exception as e:
        print(f"获取嵌入向量时出错: {e}")
        return [0] * 768  # 出错时返回零向量


async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """处理单个文本块。"""
    # 提取标题和摘要
    extracted = await get_title_and_summary(chunk, url)
    
    # 获取文本块的嵌入向量
    embedding = await get_embedding(chunk)
    
    # 创建元数据
    metadata = {
        "source": "crawl4ai_docs",  # 数据源
        "chunk_size": len(chunk),  # 文本块的长度
        "crawled_at": datetime.now(timezone.utc).isoformat(),  # 爬取时间，使用UTC时区并以ISO格式存储
        "url_path": urlparse(url).path  # 从URL中提取的路径部分
    }
    
    # 返回处理后的文本块对象
    return ProcessedChunk(
        url=url,  # 原始文本块所在的网页URL
        chunk_number=chunk_number,  # 文本块的顺序编号
        title=extracted['title'],  # 提取的标题
        summary=extracted['summary'],  # 提取的摘要
        content=chunk,  # 原始文本块的内容
        metadata=metadata,  # 元数据字典
        embedding=embedding  # 文本块的嵌入向量
    )


async def insert_chunk(chunk: ProcessedChunk):
    """将处理后的文本块插入到Supabase中。"""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        if result.data:
            print(f"插入文本块 {chunk.chunk_number} for {chunk.url}")
        else:
            print(f"插入失败: {chunk.url} - {result}")
        return result
    except Exception as e:
        print(f"插入文本块时出错: {e}")
        return None


async def process_and_store_document(url: str, markdown: str):
    """处理文档并将文本块并行存储。"""
    # 将文档分割成文本块
    chunks = chunk_text(markdown)
    
    # 并行处理文本块
    tasks = [
        process_chunk(chunk, i, url)  # 创建处理文本块的任务
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)  # 等待所有处理任务完成
    
    # 并行存储处理后的文本块
    insert_tasks = [
        insert_chunk(chunk)  # 创建存储文本块的任务
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)  # 等待所有存储任务完成


async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """并行爬取多个URL，并限制并发数量。"""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"成功爬取: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"失败: {url} - 错误: {result.error_message}")
        
        # 添加进度提示
        total_urls = len(urls)
        for i, url in enumerate(urls):
            print(f"正在处理 URL {i + 1}/{total_urls}: {url}")
            await process_url(url)
    finally:
        await crawler.close()


def get_docs_urls() -> List[str]:
    """从文档sitemap中获取URL。"""
    sitemap_url = "https://docs.crawl4ai.com/sitemap.xml"  # 爬取不同的文档时要修改此处的URL
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # 解析XML
        root = ElementTree.fromstring(response.content)
        
        # 从sitemap中提取所有URL
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"获取sitemap时出错: {e}")
        return []

async def main():
    # 从文档中获取URL
    urls = get_docs_urls()
    if not urls:
        print("没有找到要爬取的URL")
        return
    
    print(f"找到 {len(urls)} 个URL要爬取")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())