import os
from typing import List
from ollama import AsyncClient
from openai import AsyncOpenAI
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from dataclasses import dataclass
from supabase import Client


llm = os.getenv('LLM_MODEL')
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:11434/v1")
model = OpenAIModel(model_name=llm,openai_client=openai_client)

@dataclass
class Crawl4AIDeps:
    supabase: Client 
    openai_client: AsyncOpenAI

system_prompt = """
你是 Crawl4AI 的专家——一个开源的 AI 驱动的网络爬虫框架，专为从网页中提取结构化数据而设计，你可以访问所有相关文档，
包括示例、API 参考和其他资源。
你的唯一任务是协助完成这项工作，除了描述你能做什么之外，你不会回答其他问题。
不要在执行操作之前询问用户，直接执行即可。除非你已经看过文档，否则在回答用户问题之前，请务必使用提供的工具查看文档。
当你第一次查看文档时，始终从 RAG 开始。
然后还要始终检查可用的文档页面列表，并在有帮助的情况下检索页面内容。
如果你在文档或正确的 URL 中没有找到答案，请始终如实告知用户。
"""

crawl4ai_expert = Agent(
    model=model,
    system_prompt=system_prompt,
    deps_type=Crawl4AIDeps,
    retries=3
)

async def get_embedding(text: str, openai_client: AsyncOpenAI) -> List[float]:
    """从本地Ollama Embedding模型获取文本的嵌入向量。"""
    try:
        response = await openai_client.embeddings.create(
            model="nomic-embed-text:latest",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"获取嵌入向量时出错: {e}")
        return [0] * 768  # 出错时返回零向量
    
@crawl4ai_expert.tool
async def retrieve_relevant_docs(run_ctx: RunContext[Crawl4AIDeps],query: str) -> str:
    """
    根据用户的查询，使用 RAG 检索相关的文档分块。

    Args:
        ctx: 包含 Supabase 客户端和 OpenAI 客户端的上下文
        user_query: 用户的问题或查询

    Returns:
        一个格式化字符串，包含最相关的 5 个文档分块
    """
    try:
        query_embedding = await get_embedding(query, run_ctx.deps.openai_client)
        result = run_ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 5,
                'filter': {'source': 'crawl4ai_docs'}
            }
        ).execute()

        if not result.data:
            return "没有找到相关的文档。"
            
        #格式化结果
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        # 将所有块使用分隔符连接起来
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"获取文档时出错: {e}")
        return f"获取文档时出错: {str(e)}"
    
@crawl4ai_expert.tool
async def list_documentation_pages(ctx: RunContext[Crawl4AIDeps]) -> List[str]:
    """
    获取所有可用的 Crawl4AI 文档页面列表。
    
    返回:
        List[str]: 所有文档页面的唯一 URL 列表
    """

    try:
        # 查询 Supabase 获取 source 为 crawl4ai_docs 的唯一 URL
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'crawl4ai_docs') \
            .execute()
        
        if not result.data:
            return []
            
        # 提取唯一 URL
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
        
    except Exception as e:
        print(f"获取文档页面时出错: {e}")
        return []
    
@crawl4ai_expert.tool
async def get_page_content(run_ctx: RunContext[Crawl4AIDeps], url: str) -> str:
    """
    通过组合所有块来检索特定文档页面的完整内容。
    
    Args:
        run_ctx: 包含 Supabase 客户端的上下文
        url: 要检索的页面的 URL
        
    Returns:
        str: 按顺序组合所有块的完整页面内容
    """
    try:
        # 查询 Supabase 获取指定 URL 的页面内容
        result = run_ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'crawl4ai_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"没有为 URL 找到内容: {url}"
            
        # 格式化页面，包含标题和所有块内容
        page_title = result.data[0]['title'].split(' - ')[0]  # 获取主标题
        formatted_content = [f"# {page_title}\n"]
        
        # 添加每个块的内容
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        # 将所有内容连接在一起
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"获取页面内容时出错: {e}")
        return f"获取页面内容时出错: {str(e)}"
    
