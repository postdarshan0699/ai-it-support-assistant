"""
Streamlit UI — run with: streamlit run app.py

Keeps the full AgentState in st.session_state so employee_id and
conversation history persist across turns, which is what lets the agent
handle a flow like: "Ask employee ID -> confirm -> check tickets -> respond"
without re-asking every time.
"""

import streamlit as st
from src.graph.build_graph import build_graph

st.set_page_config(page_title="IT Support Assistant", page_icon="\U0001F4BB")
st.title("IT Support Assistant")
st.caption("Agentic AI \u2014 LangGraph routing across Knowledge Search, Ticket Lookup, and Ticket Creation")

if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "state" not in st.session_state:
    st.session_state.state = {
        "messages": [],
        "user_query": "",
        "intent": None,
        "employee_id": None,
        "issue_description": None,
        "tool_result": None,
        "final_response": None,
    }

with st.sidebar:
    st.subheader("Session state")
    st.write("Employee ID:", st.session_state.state.get("employee_id") or "\u2014")
    st.write("Last intent:", st.session_state.state.get("intent") or "\u2014")
    if st.button("Reset conversation"):
        st.session_state.state = {
            "messages": [], "user_query": "", "intent": None, "employee_id": None,
            "issue_description": None, "tool_result": None, "final_response": None,
        }
        st.rerun()

for msg in st.session_state.state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Describe your IT issue, or ask about an existing ticket...")

if user_input:
    st.session_state.state["messages"].append({"role": "user", "content": user_input})
    st.session_state.state["user_query"] = user_input

    with st.chat_message("user"):
        st.write(user_input)

    with st.spinner("Working on it..."):
        try:
            result = st.session_state.graph.invoke(st.session_state.state)
            st.session_state.state.update(result)
        except Exception as exc:
            st.session_state.state["final_response"] = f"Something went wrong: {exc}"

    reply = st.session_state.state.get("final_response", "Sorry, I couldn't process that.")
    st.session_state.state["messages"].append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.write(reply)
        if st.session_state.state.get("intent"):
            st.caption(f"Routed to: {st.session_state.state['intent']}")
