from langchain_core.tools import tool
from langchain_tavily import TavilySearch
import os
from .model import llm

tavily_api_key = os.getenv("TAVILY_API_KEY")


@tool("web_search")
def web_search(query: str) -> str:
    """
    Perform a real-time web search.
    
    Args:
        query (str): The search query.
    """
    try:
        search = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)
        results = search.invoke(query)
        formatted_results = "\n".join([f"- {r['title']}: {r['content'][:200]}..." for r in results])
        return formatted_results if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


@tool("summarizer")
def summarizer(query: str) -> str:
    """
    Summarize any text or topic using the LLM.
    Args:
        query (str): The text or topic to summarize.
    """ 
    try:
        prompt = f"Summarize the following text in 3-4 sentences:\n\n{query}"
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Summary error: {str(e)}"