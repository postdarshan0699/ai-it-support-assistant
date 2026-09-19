# AI IT Support Assistant — Agentic AI with LangGraph

**Capstone Project 3 — IIT Patna GenAI Program**

## 1. Problem Statement

In a company, employees usually contact IT support for different types of problems.

For example:

* "How do I reset my VPN password?"
* "What is the status of my existing ticket?"
* "My printer is not working. Can you create a ticket?"

Normally, these requests may have to be handled differently depending on what the employee needs.

For this project, I built an AI-based IT Support Assistant that can understand what the employee is asking and decide what action needs to be taken.

The assistant can:

1. Search the IT knowledge base for common problems.
2. Check the status of existing tickets.
3. Create a new support ticket.
4. Ask the user for missing information when required.

The assistant also remembers information provided earlier in the conversation. For example, if the employee gives their employee ID in one message, they don't have to enter it again in the next message.

---

## 2. How the Application Works

The application starts with the user's message and first tries to understand what the user wants.

The `classify_intent` node uses the LLM to identify the type of request and extract information such as the employee ID and issue description.

Based on the intent, LangGraph routes the request to the appropriate node:

```text
User Message
     |
     v
classify_intent
     |
     |---- knowledge_search
     |          |
     |          v
     |      Vector Store
     |
     |---- ticket_lookup
     |
     |---- ticket_creation
     |
     |---- clarify
     |
     v
generate_response
     |
     v
    END
```

For knowledge-related questions, the assistant searches the knowledge base using a **vector store**. The knowledge articles are converted into embeddings and stored as vectors. When a user asks a question, the question is also converted into an embedding and the most relevant articles are retrieved based on similarity.

This means the assistant can find relevant information even when the user's wording is different from the wording used in the knowledge base.

For example, the knowledge base might contain:

```text
How to reset your VPN password
```

A user could ask:

```text
I forgot my VPN password. How can I change it?
```

The vector search can still retrieve the relevant article because the meaning of the two requests is similar.

For ticket-related requests, the agent uses the ticket lookup and ticket creation tools instead of the vector store.

The application also maintains a shared `AgentState`. This allows information such as the employee ID, intent, issue description, and tool results to be carried through the conversation.

---

## 3. Project Structure

```
ai-it-support-assistant/
├── app.py                     # Streamlit UI — entry point
├── config.py                  # Loads settings and builds the LLM/embeddings clients
├── requirements.txt
├── runtime.txt                # Pins the Python version for Streamlit deployment
├── .env.example                # Template for local API keys (never the real keys)
├── data/
│   ├── knowledge_base.json    # IT knowledge base articles
│   └── tickets.json           # Sample tickets, used as a lightweight ticket "database"
└── src/
    ├── db.py                  # Reads and writes the JSON ticket data
    ├── tools/
    │   ├── knowledge_search.py   # Tool 1 — vector search over the knowledge base
    │   ├── vector_store.py       # Builds the FAISS index used by knowledge_search
    │   ├── ticket_lookup.py      # Tool 2 — finds existing tickets
    │   └── ticket_creation.py    # Tool 3 — creates a new ticket, with a duplicate check
    └── graph/
        ├── state.py            # Defines the shared AgentState
        ├── schemas.py           # Pydantic schema for the intent classification output
        └── build_graph.py       # Builds the LangGraph — nodes, edges, and routing
```

I kept the tools, the graph logic, and the UI in separate files on purpose. It made it much easier to test one tool at a time without needing to run the whole Streamlit app, and it matches the "modular code" requirement from the brief.

## 4. Setup

**1. Clone the repository and create a virtual environment**
```bash
git clone https://github.com/postdarshan0699/ai-it-support-assistant.git
cd ai-it-support-assistant
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add API keys**

For local development, copy `.env.example` to `.env` and fill in your own key:
```
LLM_PROVIDER=gemini
GOOGLE_API_KEY=your-key-here
GOOGLE_MODEL=gemini-2.5-flash
```
`.env` is listed in `.gitignore` and is never committed — only `.env.example`, which has no real keys in it, is tracked.

For the deployed version, the same values are stored in Streamlit's own secrets manager instead of a `.env` file, so the key never has to exist in the repository at all.

## 5. How to Run

```bash
streamlit run app.py
```
This opens the chat interface at `http://localhost:8501`. The deployed version runs the same `app.py` on Streamlit Community Cloud, reading its secrets from Streamlit's dashboard instead of a local file.

## 6. Sample I/O

**Input:** *"My printer isn't working, please log a ticket."*
**Agent:** *"Could you provide your employee ID? I need that before I can help further."*
**Input:** *"EMP9999"*
**Agent:** *"I've created ticket TCK-1004 for your printer issue. It's currently marked as Open."*

**Input:** *"I forgot my VPN password, how do I change it?"*
**Agent:** retrieves the "How to reset your VPN password" article from the vector store and explains the reset steps, even though the wording doesn't match the article title.

## 7. The Three Tools

The agent has three tools available, and `classify_intent` decides which one to call:

- **Knowledge Search** — searches the IT knowledge base using the vector store described in Section 10. Used for "how do I..." style questions.
- **Ticket Lookup** — checks `data/tickets.json` for existing tickets, filtered by employee ID and/or issue keyword. Used for "what's the status of..." questions.
- **Ticket Creation** — creates a new ticket after checking the employee's open tickets for a similar existing issue, so the same problem doesn't get logged twice.

Each tool is a plain Python function with no LLM call inside it — the LLM's only job is deciding *which* tool to use, not doing the tool's work itself.

---

## 8. Technologies Used

The main technologies used in this project are:

* **Python** — main programming language
* **LangGraph** — manages the agent workflow and routing
* **LangChain** — used for LLM, tools, embeddings, and retrieval
* **Pydantic** — validates the structured intent output
* **OpenAI / Gemini** — LLM providers
* **Embeddings** — used to convert knowledge articles and user queries into vectors
* **Vector Store** — stores the knowledge-base embeddings and retrieves relevant documents
* **Streamlit** — user interface
* **JSON** — used for the ticket data and source knowledge-base data

The vector store is used mainly for the knowledge-search part of the application, while the ticket information is handled separately.

---

## 9. Why I Used LangGraph

The main reason for using LangGraph is that the application has multiple possible paths depending on what the user wants.

For example, a user asking:

```text
How do I reset my VPN password?
```

doesn't need a ticket to be created. The request can be sent to the knowledge-search node, which retrieves the relevant information from the vector store.

On the other hand:

```text
My laptop is not working. Please create a ticket.
```

needs to go through the ticket-creation flow.

The graph makes these different paths easier to manage.

It also allows the application to keep state between steps. For example, if the employee ID is missing during ticket creation, the assistant can ask for it and continue the process once the user provides it.

---

## 10. Knowledge Search Using Vector Store

The knowledge base contains IT support articles covering common issues such as VPN problems, Wi-Fi, software installation, email, printers, security, and hardware.

Before searching, the knowledge-base content is converted into embeddings and stored in the vector store.

When the user asks a question:

1. The user's question is converted into an embedding.
2. The vector store searches for similar content.
3. The most relevant knowledge-base articles are returned.
4. The retrieved information is passed back to the agent.
5. The LLM uses the retrieved information to generate the final response.

This makes the knowledge search more flexible than simple keyword matching.

For example, these questions can potentially retrieve the same article:

```text
How do I reset my VPN password?

I forgot my VPN password.

I can't remember my VPN password. How can I change it?
```

Even though the wording is different, they are asking about the same underlying problem.

---

## 11. Current Limitations

There are still some limitations in the current version.

### Vector search

The vector store improves the knowledge search compared with simple keyword matching, but the quality of the results still depends on the embedding model, document chunks, and similarity settings.

The knowledge base is also relatively small because this is a capstone project.

### Data storage

The tickets are currently stored in JSON files.

This keeps the project simple, but a real application with multiple users would need a proper database such as SQLite, PostgreSQL, or another production database.

### Duplicate detection

The ticket duplicate check currently uses a simple similarity/word-overlap approach.

It could be improved by using the same type of semantic similarity approach used for the knowledge search.

### Authentication

The application currently does not have real employee authentication.

The employee ID is provided by the user, so a production system should connect this to the company's authentication or employee directory.

---

## 12. Possible Improvements

If I continue developing the project, some improvements I would like to make are:

* Improve the document chunking and retrieval process.
* Experiment with different embedding models.
* Add a reranker to improve retrieval accuracy.
* Replace JSON ticket storage with a proper database.
* Add employee authentication.
* Use semantic similarity for duplicate-ticket detection.
* Add more IT support tools.
* Add better error handling and logging.
* Add evaluation tests for the retrieval and agent decisions.
* Deploy the application so multiple users can access it.

---

## 13. Conclusion

This project demonstrates how an LLM, LangGraph, tools, and a vector store can be combined to create a simple IT support agent.

The LLM is not being used only to generate a response. It also helps understand the user's request and decide which part of the application should handle it.

For knowledge-related questions, the vector store helps retrieve relevant information from the IT knowledge base. For ticket-related requests, the agent can look up existing tickets or create a new one when required.

The current version is a working prototype, but the same approach can be extended with a larger knowledge base, better retrieval, authentication, a proper database, and additional IT support tools.
