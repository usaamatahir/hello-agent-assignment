from load_dotenv import load_dotenv
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncOpenAI
import os


load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"), base_url=os.getenv("BASE_URL")
)
set_default_openai_client(client)

llm_model = OpenAIChatCompletionsModel(
    model=os.getenv("MODEL_NAME"),
    openai_client=client,
)

set_tracing_disabled(disabled=True)
agent = Agent(
    name="hello-agent",
    instructions="You are a helpful assistant that greets the user in short sentence.",
    model=llm_model,
)


result = Runner.run_sync(agent, "Say hello to the user")


print(result.final_output)
