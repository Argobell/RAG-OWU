import os
import sys
import psutil
import asyncio
import requests
from xml.etree import ElementTree

__location__ = os.path.dirname(os.path.abspath(__file__))
__output__ = os.path.join(__location__, "output")

# 将父目录添加到系统路径中
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from typing import List
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    print("\n=== 并行抓取带浏览器复用 + 内存监控 ===")

    # 我们将跟踪所有任务的峰值内存使用情况
    peak_memory = 0
    process = psutil.Process(os.getpid())

    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # 以字节为单位
        if current_mem > peak_memory:
            peak_memory = current_mem
        print(f"{prefix} 当前内存: {current_mem // (1024 * 1024)} MB, 峰值: {peak_memory // (1024 * 1024)} MB")

    # 最小化浏览器配置
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,   # 修正为 'verbose=False'
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # 创建抓取器实例
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # 我们将URL分成多个批次，每个批次的数量由 'max_concurrent' 决定
        success_count = 0
        fail_count = 0
        for i in range(0, len(urls), max_concurrent):
            batch = urls[i : i + max_concurrent]
            tasks = []

            for j, url in enumerate(batch):
                # 每个并发子任务一个唯一的 session_id
                session_id = f"parallel_session_{i + j}"
                task = crawler.arun(url=url, config=crawl_config, session_id=session_id)
                tasks.append(task)

            # 在启动任务之前检查内存使用情况
            log_memory(prefix=f"启动批次 {i//max_concurrent + 1} 之前: ")

            # 收集结果
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 在任务完成后检查内存使用情况
            log_memory(prefix=f"完成批次 {i//max_concurrent + 1} 之后: ")

            # 评估结果
            for url, result in zip(batch, results):
                if isinstance(result, Exception):
                    print(f"抓取 {url} 时出错: {result}")
                    fail_count += 1
                elif result.success:
                    success_count += 1
                else:
                    fail_count += 1

        print(f"\n摘要:")
        print(f"  - 成功抓取: {success_count}")
        print(f"  - 失败: {fail_count}")

    finally:
        print("\n关闭抓取器...")
        await crawler.close()
        # 最终内存日志
        log_memory(prefix="最终: ")
        print(f"\n峰值内存使用 (MB): {peak_memory // (1024 * 1024)}")

def get_docs_urls():
    """
    从想要爬取的文档中获取所有URL。
    使用sitemap (https://docs.crawl4ai.com/sitemap.xml) 来获取这些URL。
    
    返回:
        List[str]: URL列表
    """            
    sitemap_url = "https://docs.crawl4ai.com/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        # 解析XML
        root = ElementTree.fromstring(response.content)
        
        # 从sitemap提取所有URL
        # 命名空间通常在根元素中定义
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        
        return urls
    except Exception as e:
        print(f"获取sitemap时出错: {e}")
        return []        

async def main():
    urls = get_docs_urls()
    if urls:
        print(f"找到 {len(urls)} 个URL进行抓取")
        await crawl_parallel(urls, max_concurrent=10)
    else:
        print("没有找到任何URL进行抓取")    

if __name__ == "__main__":
    asyncio.run(main())

