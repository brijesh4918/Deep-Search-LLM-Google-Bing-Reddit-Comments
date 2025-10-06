from dotenv import load_dotenv
from typing import Annotated, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model

# Local module imports
from web_operations import serp_search, reddit_search_api, reddit_post_retrieval
from prompts import (
    get_reddit_analysis_messages,
    get_google_analysis_messages,
    get_bing_analysis_messages,
    get_reddit_url_analysis_messages,
    get_synthesis_messages
)

# -------------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------------
load_dotenv()
llm = init_chat_model("gpt-4o")

# -------------------------------------------------------------------------
# State Definition
# -------------------------------------------------------------------------
class ResearchState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str | None
    google_output: str | None
    bing_output: str | None
    reddit_output: str | None
    chosen_reddit_urls: list[str] | None
    reddit_posts_data: list | None
    google_summary: str | None
    bing_summary: str | None
    reddit_summary: str | None
    final_summary: str | None


class RedditURLSelection(BaseModel):
    selected_urls: List[str] = Field(
        description="Top Reddit URLs containing meaningful insights or discussions related to the query."
    )

# -------------------------------------------------------------------------
# Core Search Nodes
# -------------------------------------------------------------------------
def perform_google_search(state: ResearchState):
    query = state.get("user_query", "")
    print(f"[Google] Searching for: {query}")
    return {"google_output": serp_search(query, engine="google")}


def perform_bing_search(state: ResearchState):
    query = state.get("user_query", "")
    print(f"[Bing] Searching for: {query}")
    return {"bing_output": serp_search(query, engine="bing")}


def perform_reddit_search(state: ResearchState):
    query = state.get("user_query", "")
    print(f"[Reddit] Searching for: {query}")
    reddit_data = reddit_search_api(keyword=query)
    print(reddit_data)
    return {"reddit_output": reddit_data}

# -------------------------------------------------------------------------
# Reddit URL Filtering and Post Retrieval
# -------------------------------------------------------------------------
def extract_reddit_urls(state: ResearchState):
    query = state.get("user_query", "")
    reddit_raw = state.get("reddit_output", "")

    if not reddit_raw:
        return {"chosen_reddit_urls": []}

    structured_llm = llm.with_structured_output(RedditURLSelection)
    prompts = get_reddit_url_analysis_messages(query, reddit_raw)

    try:
        result = structured_llm.invoke(prompts)
        urls = result.selected_urls
        print("🧩 Selected Reddit URLs:")
        for idx, u in enumerate(urls, 1):
            print(f"  {idx}. {u}")
    except Exception as e:
        print(f"Error during Reddit URL extraction: {e}")
        urls = []

    return {"chosen_reddit_urls": urls}


def fetch_reddit_posts(state: ResearchState):
    urls = state.get("chosen_reddit_urls", [])
    if not urls:
        print("[Reddit] No URLs to fetch.")
        return {"reddit_posts_data": []}

    print(f"[Reddit] Fetching posts from {len(urls)} URLs...")
    posts = reddit_post_retrieval(urls)

    if posts:
        print(f"[Reddit] Successfully retrieved {len(posts)} posts.")
    else:
        print("[Reddit] Failed to retrieve any post data.")

    return {"reddit_posts_data": posts or []}

# -------------------------------------------------------------------------
# Analysis Nodes (Google, Bing, Reddit)
# -------------------------------------------------------------------------
def analyze_google_data(state: ResearchState):
    print("[Analysis] Evaluating Google search results...")
    query = state.get("user_query", "")
    google_content = state.get("google_output", "")
    prompts = get_google_analysis_messages(query, google_content)
    response = llm.invoke(prompts)
    return {"google_summary": response.content}


def analyze_bing_data(state: ResearchState):
    print("[Analysis] Evaluating Bing search results...")
    query = state.get("user_query", "")
    bing_content = state.get("bing_output", "")
    prompts = get_bing_analysis_messages(query, bing_content)
    response = llm.invoke(prompts)
    return {"bing_summary": response.content}


def analyze_reddit_data(state: ResearchState):
    print("[Analysis] Evaluating Reddit discussions...")
    query = state.get("user_query", "")
    reddit_raw = state.get("reddit_output", "")
    reddit_posts = state.get("reddit_posts_data", "")
    prompts = get_reddit_analysis_messages(query, reddit_raw, reddit_posts)
    response = llm.invoke(prompts)
    return {"reddit_summary": response.content}

# -------------------------------------------------------------------------
# Final Synthesis
# -------------------------------------------------------------------------
def combine_insights(state: ResearchState):
    print("[Synthesis] Merging all analyses into a unified summary...")
    query = state.get("user_query", "")
    google_summary = state.get("google_summary", "")
    bing_summary = state.get("bing_summary", "")
    reddit_summary = state.get("reddit_summary", "")

    prompts = get_synthesis_messages(query, google_summary, bing_summary, reddit_summary)
    response = llm.invoke(prompts)
    final = response.content

    return {
        "final_summary": final,
        "messages": [{"role": "assistant", "content": final}],
    }

# -------------------------------------------------------------------------
# Graph Construction
# -------------------------------------------------------------------------
workflow = StateGraph(ResearchState)

workflow.add_node("google_search", perform_google_search)
workflow.add_node("bing_search", perform_bing_search)
workflow.add_node("reddit_search", perform_reddit_search)
workflow.add_node("extract_reddit_urls", extract_reddit_urls)
workflow.add_node("fetch_reddit_posts", fetch_reddit_posts)
workflow.add_node("analyze_google", analyze_google_data)
workflow.add_node("analyze_bing", analyze_bing_data)
workflow.add_node("analyze_reddit", analyze_reddit_data)
workflow.add_node("synthesize_results", combine_insights)

# Edges
workflow.add_edge(START, "google_search")
workflow.add_edge(START, "bing_search")
workflow.add_edge(START, "reddit_search")

workflow.add_edge("google_search", "extract_reddit_urls")
workflow.add_edge("bing_search", "extract_reddit_urls")
workflow.add_edge("reddit_search", "extract_reddit_urls")
workflow.add_edge("extract_reddit_urls", "fetch_reddit_posts")

workflow.add_edge("fetch_reddit_posts", "analyze_google")
workflow.add_edge("fetch_reddit_posts", "analyze_bing")
workflow.add_edge("fetch_reddit_posts", "analyze_reddit")

workflow.add_edge("analyze_google", "synthesize_results")
workflow.add_edge("analyze_bing", "synthesize_results")
workflow.add_edge("analyze_reddit", "synthesize_results")

workflow.add_edge("synthesize_results", END)

compiled_graph = workflow.compile()

# -------------------------------------------------------------------------
# CLI Execution
# -------------------------------------------------------------------------
def start_agent():
    print("🤖 Multi-Source Deep Research Agent")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Enter your question: ")
        if query.lower() == "exit":
            print("Goodbye 👋")
            break

        print("\n🔎 Starting comprehensive web research...\n")
        state = {
            "messages": [{"role": "user", "content": query}],
            "user_query": query,
            "google_output": None,
            "bing_output": None,
            "reddit_output": None,
            "chosen_reddit_urls": None,
            "reddit_posts_data": None,
            "google_summary": None,
            "bing_summary": None,
            "reddit_summary": None,
            "final_summary": None,
        }

        final_state = compiled_graph.invoke(state)

        if final_state.get("final_summary"):
            print("\n🧠 Final Synthesized Insight:\n")
            print(final_state["final_summary"])
            print("\n" + "-" * 80)


if __name__ == "__main__":
    start_agent()