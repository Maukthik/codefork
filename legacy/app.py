import streamlit as st
import agent  # Imports your existing agent.py logic

st.set_page_config(page_title="AI Coding Agent", layout="wide")
st.title("Autonomous Dev Agent 🤖")
st.markdown("A secure, Docker-sandboxed ReAct loop for code generation.")

task = st.text_area("Enter a coding task for the agent:")

if st.button("Execute Task"):
    if task:
        with st.spinner("Agent is reasoning, writing, and testing code in Docker..."):
            # Triggers your existing terminal logic in the background
            agent.run_agent(task)
        st.success("Task Complete! Check your local workspace folder for the generated files.")
    else:
        st.warning("Please enter a task.")