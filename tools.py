import os
os.environ['USE_TF'] = '0'

import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_core.tools import tool, StructuredTool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from supabase import create_client
from tavily import TavilyClient

load_dotenv()

# ---- Shared clients ----
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)


# ================= PDF RAG tool =================

@tool
def pdf_rag_tool(question: str) -> str:
    """Answer questions about Lionsgate's quarterly earnings using the PDF-RAG knowledge base (Supabase pgvector)."""
    query_vector = embeddings.embed_query(question)
    result = supabase.rpc("match_documents", {
        "query_embedding": query_vector,
        "match_count": 3
    }).execute()

    context = "\n\n---\n\n".join(row["content"] for row in result.data)

    prompt = f"""Answer the question using only the context below. If the answer isn't in the context, say so.

Context:
{context}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content


# ================= SQL agent tool =================

class ImdbQuery(BaseModel):
    genre: Optional[str] = Field(None, description="Genre to filter by, e.g. Horror, Drama")
    min_votes: Optional[int] = Field(None, description="Minimum number of votes")
    year_from: Optional[int] = Field(None, description="Earliest release year")
    year_to: Optional[int] = Field(None, description="Latest release year")

DF = pd.read_csv("imdb_filtered_2000_2024.csv")
DF["startYear"] = DF["startYear"].fillna(0).astype(int)
DF["numVotes"] = DF["numVotes"].fillna(0).astype(int)

def imdb_lookup(genre: Optional[str] = None, min_votes: Optional[int] = None,
                 year_from: Optional[int] = None, year_to: Optional[int] = None) -> str:
    df = DF.copy()
    if genre:
        df = df[df["genres"].str.contains(genre, case=False, na=False)]
    if min_votes:
        df = df[df["numVotes"] >= min_votes]
    if year_from:
        df = df[df["startYear"] >= year_from]
    if year_to:
        df = df[df["startYear"] <= year_to]
    if df.empty:
        return "No movies found matching those filters."
    df = df.sort_values("averageRating", ascending=False).head(10)
    lines = [
        f"{r['primaryTitle']} ({r['startYear']}) - Rating {r['averageRating']}, Votes {r['numVotes']}"
        for _, r in df.iterrows()
    ]
    return "\n".join(lines)

imdb_tool = StructuredTool.from_function(
    func=imdb_lookup,
    name="imdb_lookup",
    description="Query a dataset of 30,000+ movies (2000-2024) by genre, vote count, or year range.",
    args_schema=ImdbQuery,
)

db = SQLDatabase.from_uri("sqlite:///imdb.db")
sql_tools = SQLDatabaseToolkit(db=db, llm=llm).get_tools()

SQL_SYSTEM_INSTRUCTION = "You are a movie research assistant with access to an IMDb dataset and SQL database."

def get_sql_agent():
    return create_react_agent(model=llm, tools=[imdb_tool] + sql_tools, prompt=SQL_SYSTEM_INSTRUCTION)

@tool
def sql_agent_tool(question: str) -> str:
    """Answer questions about IMDb movie data (ratings, votes, genres, release years) using SQL + a filtered dataset."""
    agent_executor = get_sql_agent()
    final_content = ""
    for event in agent_executor.stream(
        {"messages": [("human", question)]},
        stream_mode="values",
        config={"recursion_limit": 50},
    ):
        if "messages" in event:
            last_msg = event["messages"][-1]
            if last_msg.type == "ai" and not getattr(last_msg, "tool_calls", None):
                final_content = last_msg.content
    return final_content


# ================= Web search tool =================

@tool
def web_search_tool(query: str) -> str:
    """Search the web for current, real-world, or time-sensitive information not in the SQL database or PDF (e.g. breaking news, recent earnings)."""
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return "Tavily API key not configured."
    try:
        client = TavilyClient(api_key=key)
        results = client.search(query, max_results=4)
        chunks = []
        for r in results.get("results", []):
            chunks.append(f"{r.get('title','Untitled')}: {r.get('content','')[:300]}")
        return "\n\n".join(chunks) if chunks else "No results found."
    except Exception as e:
        return f"Search failed: {e}"