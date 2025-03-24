import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from src.langgraphagenticai.state.state import State
import json


class DisplayResultStreamlit:
    def __init__(self, usecase, graph, user_message):
        self.usecase = usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        usecase = self.usecase
        graph = self.graph
        user_message = self.user_message
        
        if usecase == "Basic Chatbot":
            for event in graph.stream({'messages': ("user", user_message)}):
                print(event.values())
                for value in event.values():
                    print(value['messages'])
                    with st.chat_message("user"):
                        st.write(user_message)
                    with st.chat_message("assistant"):
                        st.write(value["messages"].content)
        
        elif usecase == "Chatbot with Tool":
            # Prepare state and invoke the graph
            initial_state = {"messages": [user_message]}
            res = graph.invoke(initial_state)
            for message in res['messages']:
                if isinstance(message, HumanMessage):
                    with st.chat_message("user"):
                        st.write(message.content)
                elif isinstance(message, ToolMessage):
                    with st.chat_message("ai"):
                        st.write("Tool Call Start")
                        st.write(message.content)
                        st.write("Tool Call End")
                elif isinstance(message, AIMessage) and message.content:
                    with st.chat_message("assistant"):
                        st.write(message.content)
        
        elif usecase == "Orchestrator & Synthesizer":
            st.title("📄⚙️ Orchestrator & Synthesizer")

            links = user_message.split("\n")
            if links:
                # ✅ Use State instead of OrchestrationState
                initial_state = State(
                    messages=[HumanMessage(content="\n".join(links))],
                    fetched_data={},
                    processed_data={},
                    final_report="",
                    user_feedback=""
                )

                # ✅ Invoke the workflow graph
                final_state = graph.invoke(initial_state)
                st.write(final_state)
                # ✅ Display intermediate and final results
                st.subheader("📥 Fetched Data")
                st.write(final_state["fetched_data"])

                st.subheader("🔄 Processed Data")
                st.write(final_state["processed_data"])

                st.subheader("📑 Synthesized Report")
                st.write(final_state["final_report"])

                # ✅ Collect feedback and refine the report
                feedback = st.text_area("✍️ Provide feedback to refine the report:")
                if st.button("🔄 Submit Feedback"):
                    refined_state = graph.invoke(
                        State(
                            messages=[HumanMessage(content=final_state["final_report"])],
                            fetched_data=final_state["fetched_data"],
                            processed_data=final_state["processed_data"],
                            final_report=final_state["final_report"],
                            user_feedback=feedback
                        )
                    )
                    st.subheader("✅ Refined Report")
                    st.write(refined_state["final_report"])
            else:
                st.warning("⚠️ Please enter at least one link.")
