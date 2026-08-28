import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from tools import pdf_rag_tool, sql_agent_tool, web_search_tool

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

TOOLS = [pdf_rag_tool, sql_agent_tool, web_search_tool]
TOOL_MAP = {t.name: t for t in TOOLS}

PLANNING_PROMPT = """You are a planner. Given the user question below, identify EVERY distinct
piece of information being asked for, and list ALL tools needed to answer it completely.

Available tools:
- pdf_rag_tool: Lionsgate quarterly earnings PDF
- sql_agent_tool: IMDb movie database (ratings, votes, genres, years)
- web_search_tool: live web search for current/recent information

Respond with ONLY a JSON list of tool names needed, nothing else. Example: ["pdf_rag_tool", "sql_agent_tool"]

Question: {question}
"""

def run_supervisor(question: str) -> dict:
    plan_response = llm.invoke(PLANNING_PROMPT.format(question=question))
    raw = plan_response.content.strip()

    try:
        tool_names = json.loads(raw)
    except json.JSONDecodeError:
        tool_names = [name for name in TOOL_MAP if name in raw]

    tool_outputs = {}
    for name in tool_names:
        if name in TOOL_MAP:
            arg_key = "query" if name == "web_search_tool" else "question"
            tool_outputs[name] = TOOL_MAP[name].invoke({arg_key: question})

    return {
        "question": question,
        "tools_called": list(tool_outputs.keys()),
        "tool_outputs": tool_outputs,
    }


def synthesize(question: str, tool_outputs: dict) -> str:
    if not tool_outputs:
        return "I couldn't determine which source to use for this question."

    if len(tool_outputs) == 1:
        return list(tool_outputs.values())[0]

    context = "\n\n".join(
        f"[{name}]\n{output}" for name, output in tool_outputs.items()
    )
    prompt = f"""Combine the information below from multiple sources into one clear, coherent answer to the question. Note if sources disagree.

Sources:
{context}

Question: {question}

Answer:"""
    response = llm.invoke(prompt)
    return response.content


def run_pipeline(question: str) -> dict:
    result = run_supervisor(question)
    final_answer = synthesize(question, result["tool_outputs"])
    result["final_answer"] = final_answer
    return result


if __name__ == "__main__":
    q = "What was Lionsgate's revenue this quarter, and how many highly-rated horror movies came out in 2020?"
    output = run_pipeline(q)
    print("Tools called:", output["tools_called"])
    print("\nFinal answer:\n", output["final_answer"])