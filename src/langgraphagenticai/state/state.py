from typing import Annotated, Literal, Optional, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    """
    Defines the graph state, tracking messages and intermediate results.
    Supports all three workflows:
    1️⃣ Basic Chatbot
    2️⃣ Chatbot with Tools
    3️⃣ Orchestrator & Synthesizer
    """
    messages: Annotated[List[HumanMessage], add_messages]  # Chat messages
    fetched_data: Dict[str, str]  # Stores fetched content (Orchestration)
    processed_data: Dict[str, str]  # Stores processed content (Orchestration)
    final_report: str  # Stores synthesized report (Orchestration)
    user_feedback: str  # Stores user feedback


