# Multi-Agent ReAct System with Planner and Routed Executors using ToolNode

from langchain_aws import ChatBedrock
from typing import TypedDict, List, Union
from langchain.callbacks.base import BaseCallbackHandler
import boto3
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")

def rerank_segments(documents: List[str],query: str) -> List[str]:
    compressor = CrossEncoderReranker(model=rerank_model, top_n=len(documents))
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=
    # )
    doc_objs = [Document(page_content=doc) for doc in documents]
    result = compressor.compress_documents(documents=doc_objs, query=query)
    return [doc.page_content for doc in result]

    # compressed_docs = compression_retriever.invoke("What is the plan for the economy?")

def chunk_documents(doc_texts: List[str], chunk_size: int = 500, overlap: int = 50) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for doc in doc_texts:
        chunks.extend(splitter.split_text(doc))
    return chunks


MODEL_ID = 'meta.llama3-70b-instruct-v1:0'

class SimpleLogger(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        pass
        # print("\n📥 Prompt sent:", prompts)

    def on_llm_end(self, response, **kwargs):
        generations = response.generations if hasattr(response, "generations") else response
        # print("📤 Response received:", generations[0][0].text if generations else response)


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
    verbose=True
)

