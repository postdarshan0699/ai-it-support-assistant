"""
Tool 3 — Ticket Creation.
Creates a new ticket, but only after validation:
  - both employee_id and issue must be present (the graph's clarify node
    handles asking for whichever is missing before this tool is ever called)
  - no duplicate: if the same employee already has an OPEN ticket whose issue
    text overlaps significantly, we return that existing ticket instead of
    creating a new one — this is the "no duplicate tickets" safety rule from
    the capstone brief.
"""

from src.db import append_ticket
from src.tools.ticket_lookup import ticket_lookup


def _is_duplicate(employee_id: str, issue: str) -> dict | None:
    existing = ticket_lookup(employee_id=employee_id)
    issue_words = set(issue.lower().split())
    for t in existing:
        if t["status"] != "Open":
            continue
        overlap = len(issue_words & set(t["issue"].lower().split()))
        if overlap >= 3:  # simple similarity threshold
            return t
    return None


def create_ticket(employee_id: str, issue: str) -> dict:
    if not employee_id or not issue:
        raise ValueError("employee_id and issue are both required to create a ticket")

    duplicate = _is_duplicate(employee_id, issue)
    if duplicate:
        return {**duplicate, "duplicate": True}

    ticket = append_ticket(employee_id, issue)
    return {**ticket, "duplicate": False}
