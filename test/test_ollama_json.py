import asyncio
import os
import json
from ollama import AsyncClient

ollama_client = AsyncClient(host=os.getenv('OLLAMA_SERVER_URL'))

async def get_title_and_summary(chunk: str, url: str) -> dict:
    """使用Ollama提取标题和摘要。"""
    system_prompt = """你是一个AI，用于从文档块中提取标题和摘要。
    返回一个包含'title'和'summary'键的JSON对象。
    对于标题：如果看起来像是文章的开头，请提取其标题。如果是中间块，请推导一个描述性的标题。
    对于摘要：请创建一个简洁的摘要，概括该块的主要内容。
    保持标题和摘要都简洁但有信息量。"""
    
    try:
        response = await ollama_client.chat(
            model=os.getenv("LLM_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\n内容:\n{chunk}"}  # 不截取内容
            ],
            format='json'
        )
        result = json.loads(response["message"]["content"])
        if "title" not in result or "summary" not in result:
            raise ValueError("Ollama 返回的 JSON 缺少 'title' 或 'summary' 键")
        return result
    except Exception as e:
        print(f"获取标题和摘要时出错: {e}")
        return {"title": "处理标题出错", "summary": "处理摘要出错"}

async def test_get_title_and_summary():
    # 测试用的文档块和URL
    test_chunk = """# 人工智能的未来
    人工智能技术正在飞速发展，为各个行业带来了巨大变革。从制造业到医疗保健，从金融服务到零售业，AI的应用越来越广泛。"""
    test_url = "http://example.com/article"
    
    # 调用函数
    result = await get_title_and_summary(test_chunk, test_url)
    
    # 打印结果
    print(result)
    print(type(result))

# 运行测试
if __name__ == "__main__":
    asyncio.run(test_get_title_and_summary())






