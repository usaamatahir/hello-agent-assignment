from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_default_openai_api,
    set_tracing_disabled,
)
import os
from openai import AsyncOpenAI
from load_dotenv import load_dotenv


load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("BASE_URL")
)

set_default_openai_api(client)


llm_model = OpenAIChatCompletionsModel(
    model=os.getenv("MODEL_NAME"), openai_client=client
)

set_tracing_disabled(True)

agent = Agent(
    name="hello-agent-packaged",
    instructions="You are a helpful assistant, specialized in providing single-line responses.",
    model=llm_model,
)


def run_agent():
    result = Runner.run_sync(agent, "What is Agentic AI?")

    print(result.final_output)
