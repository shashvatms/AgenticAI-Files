from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

apikey = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=apikey
)

#state

class ChatState(TypedDict):
    message = Annotated[list[BaseMessage],add_messages]
#[HUmanmesg("Hey"),AImessage("hello")]

checkpointer = InMemorySaver()
def chat_node(state:ChatState):
    messages = state["message"]
    response = llm.invoke(messages)
    return {"message":[response]}

graph = StateGraph(ChatState)
graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot = graph.compile(checkpointer=checkpointer)

