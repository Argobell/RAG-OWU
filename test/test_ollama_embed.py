import os
import asyncio
from typing import List
from openai import AsyncOpenAI

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:11434/v1")

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="nomic-embed-text:latest",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 768  # Return zero vector on error
    
async def main():
    text = "Hello, world!"
    embedding = await get_embedding(text)
    print(len(embedding))
    print(embedding)

if __name__ == '__main__':
    asyncio.run(main())