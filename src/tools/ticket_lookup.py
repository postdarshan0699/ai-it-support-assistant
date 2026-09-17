"""
Tool 2 — Ticket Lookup.
Finds existing tickets by employee ID and/or a keyword match on the issue
text. Used both to answer "what's the status of my ticket" and to check for
duplicates before Tool 3 creates a new one.
"""

from src.db import load_tickets


def ticket_lookup(employee_id: str | None = None, issue_keyword: str | None = None) -> list[dict]:
    tickets = load_tickets()
    results = tickets

    if employee_id:
        results = [t for t in results if t["employee_id"].lower() == employee_id.lower()]

    if issue_keyword:
        kw = issue_keyword.lower()
        results = [t for t in results if kw in t["issue"].lower()]

    return results
