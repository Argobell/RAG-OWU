from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import streamlit as st
import json
from supabase import Client
from ollama import AsyncClient
from openai import AsyncOpenAI
# 导入所有消息部分类
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from rag_agent import crawl4ai_expert, Crawl4AIDeps
# 加载环境变量
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:11434/v1")
supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

class ChatMessage(TypedDict):
    """发送到浏览器/API 的消息格式。"""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    在 Streamlit UI 中显示消息的一部分。
    自定义如何显示系统提示、用户提示、工具调用、工具返回等。
    """
    # 系统提示
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**系统**: {part.content}")
    # 用户提示
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # 文本
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)          

async def run_agent_with_streaming(user_input: str):
    """
    使用流式文本运行代理，处理用户输入提示，
    同时在 `st.session_state.messages` 中维护整个对话。
    """
    # 准备依赖项
    deps = Crawl4AIDeps(
        supabase=supabase,
        openai_client=openai_client
    )
    # 在流中运行代理
    async with crawl4ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history= st.session_state.messages[:-1],  # 传递迄今为止的整个对话
    ) as result:
        # 我们将收集部分文本以逐步显示
        partial_text = ""
        message_placeholder = st.empty()
        # 渲染部分文本随着它的到达
        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)
        # 流结束后，我们现在有一个最终结果。
        # 添加此运行的新消息，排除用户提示消息
        filtered_messages = [msg for msg in result.new_messages() 
                            if not (hasattr(msg, 'parts') and 
                                    any(part.part_kind == 'user-prompt' for part in msg.parts))]
        st.session_state.messages.extend(filtered_messages)
        # 添加最终响应到消息中
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.title("RAG-OWU 聊天机器人")
    st.write("关于 Crawl4AI 提出任何问题。")
    # 如果会话状态中没有聊天历史，则初始化
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 显示迄今为止的对话中的所有消息
    # 每条消息要么是 ModelRequest 或 ModelResponse。
    # 我们遍历它们的部分以决定如何显示它们。
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)
    # 用户聊天输入
    user_input = st.chat_input("您对 Crawl4AI 有什么问题？")
    if user_input:
        # 我们显式地将新请求添加到对话中
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # 在 UI 中显示用户提示
        with st.chat_message("user"):
            st.markdown(user_input)
        # 在流式传输时显示助手的部分响应
        with st.chat_message("assistant"):
            # 实际运行代理，流式传输文本
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())