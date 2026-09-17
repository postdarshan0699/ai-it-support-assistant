"""
Builds the LangGraph state machine:

                         +-----------------+
                         |  classify_intent |
                         +--------+---------+
                                  |
                    conditional routing on state["intent"]
                                  |
        +---------------+--------+--------+---------------+
        |               |                 |                |
knowledge_search   ticket_lookup   ticket_creation       clarify
        |               |                 |                |
        +-------+-------+--------+--------+                |
                        |                                    |
                generate_response                            |
                        |                                    |
                       END  <----------------------------- END

`clarify` skips tool execution entirely and goes straight to a response
asking the user for the missing piece of information (e.g. employee ID) —
this is the "validates info before creating tickets" safety requirement.
"""

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from config import get_llm
from src.graph.state import AgentState
from src.graph.schemas import IntentDecision
from src.tools.knowledge_search import knowledge_search
from src.tools.ticket_lookup import ticket_lookup
from src.tools.ticket_creation import create_ticket

intent_parser = PydanticOutputParser(pydantic_object=IntentDecision)

intent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are the routing brain of an IT support assistant. Decide which tool "
     "the user's message needs. Carry forward employee_id from earlier in the "
     "conversation if the user already gave it and hasn't given a new one.\n\n"
     "{format_instructions}"),
    ("human", "Conversation so far:\n{history}\n\nKnown employee_id so far: {known_employee_id}\n\nLatest message: {user_query}"),
]).partial(format_instructions=intent_parser.get_format_instructions())

intent_chain = intent_prompt | get_llm(temperature=0) | intent_parser

response_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful, concise IT support assistant. Turn the tool result "
     "below into a natural-language reply. Never invent information that "
     "isn't in the tool result."),
    ("human", "User asked: {user_query}\n\nIntent: {intent}\nTool result:\n{tool_result}"),
])
response_chain = response_prompt | get_llm(temperature=0.3)


# ---- Nodes ----

def classify_intent_node(state: AgentState) -> dict:
    history = "\n".join(f"{m['role']}: {m['content']}" for m in state["messages"][-6:])
    decision = intent_chain.invoke({
        "history": history,
        "known_employee_id": state.get("employee_id") or "none yet",
        "user_query": state["user_query"],
    })
    return {
        "intent": decision.intent,
        "employee_id": decision.employee_id or state.get("employee_id"),
        "issue_description": decision.issue_description,
        "tool_result": {"missing_info": decision.missing_info} if decision.intent == "clarify" else None,
    }


def knowledge_search_node(state: AgentState) -> dict:
    results = knowledge_search(state["user_query"])
    return {"tool_result": {"articles": results}}


def ticket_lookup_node(state: AgentState) -> dict:
    results = ticket_lookup(employee_id=state.get("employee_id"), issue_keyword=state.get("issue_description"))
    return {"tool_result": {"tickets": results}}


def ticket_creation_node(state: AgentState) -> dict:
    try:
        ticket = create_ticket(state.get("employee_id"), state.get("issue_description"))
        return {"tool_result": {"ticket": ticket}}
    except ValueError as exc:
        return {"tool_result": {"error": str(exc)}}


def clarify_node(state: AgentState) -> dict:
    missing = (state.get("tool_result") or {}).get("missing_info", "a bit more detail")
    return {"final_response": f"Could you provide {missing}? I need that before I can help further."}


def generate_response_node(state: AgentState) -> dict:
    reply = response_chain.invoke({
        "user_query": state["user_query"],
        "intent": state["intent"],
        "tool_result": state["tool_result"],
    })
    return {"final_response": reply.content}


def route_by_intent(state: AgentState) -> str:
    return state["intent"]  # matches node names below


# ---- Build the graph ----

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("knowledge_search", knowledge_search_node)
    graph.add_node("ticket_lookup", ticket_lookup_node)
    graph.add_node("ticket_creation", ticket_creation_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("generate_response", generate_response_node)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges("classify_intent", route_by_intent, {
        "knowledge_search": "knowledge_search",
        "ticket_lookup": "ticket_lookup",
        "ticket_creation": "ticket_creation",
        "clarify": "clarify",
    })

    graph.add_edge("knowledge_search", "generate_response")
    graph.add_edge("ticket_lookup", "generate_response")
    graph.add_edge("ticket_creation", "generate_response")
    graph.add_edge("generate_response", END)
    graph.add_edge("clarify", END)

    return graph.compile()
