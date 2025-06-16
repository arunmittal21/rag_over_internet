# Multi-Agent ReAct System with Planner and Routed Executors using ToolNode

from langchain_aws import ChatBedrock
from typing import TypedDict, List, Union
from langchain.callbacks.base import BaseCallbackHandler
import boto3

MODEL_ID = 'meta.llama3-70b-instruct-v1:0'

class SimpleLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n📥 Prompt sent:", prompts)

    def on_llm_end(self, response, **kwargs):
        generations = response.generations if hasattr(response, "generations") else response
        print("📤 Response received:", generations[0][0].text if generations else response)


class AgentState(TypedDict):
    query: str
    tasks: List[dict]
    current_step: int
    results: List[Union[str, dict]]

session = boto3.Session(
    profile_name='adfs'
)
# credentials = session.get_credentials()
# print(credentials)

bedrock_runtime_client = session.client("bedrock-runtime")

llm = ChatBedrock(
    client=bedrock_runtime_client,
    model=MODEL_ID,
    temperature=0,
    callbacks=[SimpleLogger()],
)
