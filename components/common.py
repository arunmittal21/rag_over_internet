# Multi-Agent ReAct System with Planner and Routed Executors using ToolNode

from langchain_aws import ChatBedrock
from typing import TypedDict, List, Union
from langchain.callbacks.base import BaseCallbackHandler
import boto3

MODEL_ID = 'meta.llama3-70b-instruct-v1:0'

class SimpleLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        pass
        # print("\n📥 Prompt sent:", prompts)

    def on_llm_end(self, response, **kwargs):
        generations = response.generations if hasattr(response, "generations") else response
        # print("📤 Response received:", generations[0][0].text if generations else response)

        # state =  {
        #     **state,
        #     "next_task": executor_name,
        #     "task_input": executor_input,
        #     "scratchpad": scratchpad,
        #     "current_step": state["current_step"] + 1
        # }
class AgentState(TypedDict):
    query: str
    scratchpad: str
    next_task: str
    task_input: str
    result: str
    current_step: int

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
