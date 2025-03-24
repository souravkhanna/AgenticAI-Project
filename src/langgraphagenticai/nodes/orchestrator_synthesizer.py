from typing import List, Dict
import requests
from langchain_core.language_models.chat_models import BaseChatModel
from src.langgraphagenticai.state.state import State


class OrchestratorSynthesizer:
    """
    Handles orchestration of data fetching, processing, synthesizing, and refining the report.
    """

    def __init__(self, model: BaseChatModel):
        self.llm = model

    def fetch_data(self, state: State) -> State:
        """
        Fetches data from multiple links or user-pasted content.
        Stores results in a dictionary, ensuring compatibility with later functions.
        """
        sources = state.get("sources", [])  # Extract sources from state
        fetched_data = {}  # ✅ Use dictionary instead of list

        for i, source in enumerate(sources):
            if source.startswith("http"):
                try:
                    response = requests.get(source)
                    if response.status_code == 200:
                        fetched_data[f"source_{i}"] = response.text[:500]  # ✅ Store as dict
                    else:
                        fetched_data[f"source_{i}"] = f"Failed: {response.status_code}"
                except Exception as e:
                    fetched_data[f"source_{i}"] = f"Error: {str(e)}"
            else:
                fetched_data[f"source_{i}"] = f"Pasted Data: {source}"

        state["fetched_data"] = fetched_data  # ✅ Store as dictionary
        return state

    def process_data(self, state: State) -> State:
        """
        Processes fetched data to extract key insights.
        """
        fetched_data = state.get("fetched_data", {})
        processed_results = {}

        for link, content in fetched_data.items():
            processed_results[link] = self.llm.invoke([{"role": "user", "content": f"Summarize this content: {content}"}])

        # Update state with processed data
        state["processed_data"] = processed_results
        return state

    def synthesize_report(self, state: State) -> State:
        """
        Merges multiple summaries into a single cohesive report.
        """
        processed_data = state.get("processed_data", {})
        combined_text = "\n".join(processed_data.values())

        final_report = self.llm.invoke([{"role": "user", "content": f"Combine and refine into a detailed report: {combined_text}"}])

        # Update state with final report
        state["final_report"] = final_report
        return state

    def apply_feedback(self, state: State) -> State:
        """
        Adjusts the report based on human feedback.
        """
        report = state.get("final_report", "")
        user_feedback = state.get("user_feedback", "")

        refined_report = self.llm.invoke([{"role": "user", "content": f"Refine this report: {report} based on feedback: {user_feedback}"}])

        # Update state with refined report
        state["final_report"] = refined_report
        return state
