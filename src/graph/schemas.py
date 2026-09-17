from pydantic import BaseModel, Field
from typing import Literal, Optional


class IntentDecision(BaseModel):
    """What the decision node extracts from the user's message."""
    intent: Literal["knowledge_search", "ticket_lookup", "ticket_creation", "clarify"] = Field(
        description="knowledge_search for how-to/troubleshooting questions, "
                    "ticket_lookup to check existing ticket status, "
                    "ticket_creation to log a new issue, "
                    "clarify if the request is ambiguous or missing required info"
    )
    employee_id: Optional[str] = Field(default=None, description="Employee ID if mentioned, e.g. EMP2381")
    issue_description: Optional[str] = Field(default=None, description="The IT issue being described, if any")
    missing_info: Optional[str] = Field(
        default=None, description="If intent is 'clarify', what specifically is missing (e.g. 'employee ID')"
    )
