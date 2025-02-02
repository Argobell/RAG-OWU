#测试本地ollama模型启动openai api服务，并运行pydantica-ai agent服务

from pydantic_ai import Agent, RunContext
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
import os

llm = os.getenv('LLM_MODEL')
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url="http://localhost:11434/v1")
model = OpenAIModel(model_name=llm,openai_client=openai_client)


agent = Agent(  
    model=model,
    deps_type=int,
    result_type=bool,
    system_prompt=(
        'Use the `roulette_wheel` function to see if the '
        'customer has won based on the number they provide.'
    ),
    retries=3
)
@agent.tool
async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  
    """check if the square is a winner"""
    return 'winner' if square == ctx.deps else 'loser'


# Run the agent
success_number = 18  
result = agent.run_sync('Put my money on square eighteen', deps=success_number)
print(result.data)  
#> True

result = agent.run_sync('I bet five is the winner', deps=success_number)
print(result.data)