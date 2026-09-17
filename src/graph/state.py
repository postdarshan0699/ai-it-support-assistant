"""
The shared state object that flows through every node in the graph.
LangGraph passes this dict between nodes; each node returns a partial update
that gets merged in. This is what "state maintained across conversation
turns" means in practice — we keep the whole AgentState in Streamlit's
session between messages instead of rebuilding it from scratch each time.
"""

from typing import TypedDict, Optional, List, Literal


class AgentState(TypedDict):
    messages: List[dict]                 # full chat history: [{"role": "user"/"assistant", "content": str}]
    user_query: str                      # the latest user message being processed this turn
    intent: Optional[Literal["knowledge_search", "ticket_lookup", "ticket_creation", "clarify"]]
    employee_id: Optional[str]
    issue_description: Optional[str]
    tool_result: Optional[dict]          # raw output from whichever tool ran
    final_response: Optional[str]        # natural-language answer for the user
