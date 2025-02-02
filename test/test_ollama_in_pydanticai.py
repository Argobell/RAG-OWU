import os
from pydantic import BaseModel
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from dotenv import load_dotenv
load_dotenv()

class CityLocation(BaseModel):
    city: str
    country: str

client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'),base_url="http://localhost:11434/v1")

ollama_model = OpenAIModel(model_name='qwen2.5-coder:3b',openai_client=client)


agent = Agent(ollama_model, result_type=CityLocation)

result = agent.run_sync('Where were the olympics held in 2012?')
print(result.data)
#> city='London' country='United Kingdom'
print(result.usage())
"""
Usage(requests=1, request_tokens=57, response_tokens=8, total_tokens=65, details=None)
"""