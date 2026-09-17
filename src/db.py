"""
Tiny JSON-file "database" helpers. A real deployment would swap this for
SQLite (the capstone brief allows either) — kept as JSON here so the sample
data is easy to open and read directly.
"""

import json
import os
from datetime import datetime
from config import settings


def load_knowledge_base() -> list[dict]:
    with open(settings.KNOWLEDGE_BASE_PATH) as f:
        return json.load(f)


def load_tickets() -> list[dict]:
    with open(settings.TICKETS_DB_PATH) as f:
        return json.load(f)


def save_tickets(tickets: list[dict]) -> None:
    with open(settings.TICKETS_DB_PATH, "w") as f:
        json.dump(tickets, f, indent=2)


def next_ticket_id(tickets: list[dict]) -> str:
    nums = [int(t["ticket_id"].split("-")[1]) for t in tickets if t["ticket_id"].startswith("TCK-")]
    nxt = max(nums, default=1000) + 1
    return f"TCK-{nxt}"


def append_ticket(employee_id: str, issue: str) -> dict:
    tickets = load_tickets()
    new_ticket = {
        "ticket_id": next_ticket_id(tickets),
        "employee_id": employee_id,
        "issue": issue,
        "status": "Open",
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    tickets.append(new_ticket)
    save_tickets(tickets)
    return new_ticket
