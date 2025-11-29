from langgraph.prebuilt import create_react_agent
from .tools import web_search, summarizer
from .model import llm

"""
def search_agent(state):
    agent = create_react_agent(llm, [web_search])
    result = agent.invoke({"messages": state["messages"]})
    output = result["messages"][-1].content
    state["messages"].append(output)
    state["answer"] = output
    return state
"""

def search_agent(state):
    """Agent that performs web search."""
    agent = create_react_agent(llm, [web_search])
    result = agent.invoke({"messages": state["messages"]})
    last_msg = result["messages"][-1]
    output = getattr(last_msg, "content", str(last_msg))
    state["messages"].append(output)
    state["answer"] = output
    return state

"""
def summarizer_agent(state):
    agent = create_react_agent(llm, [summarizer])
    result = agent.invoke({"messages": state["messages"]})
    output = result["messages"][-1].content
    state["messages"].append(output)
    state["answer"] = output
    return state
"""

def summarizer_agent(state):
    """Agent that summarizes text."""
    agent = create_react_agent(llm, [summarizer])
    result = agent.invoke({"messages": state["messages"]})
    last_msg = result["messages"][-1]
    output = getattr(last_msg, "content", str(last_msg))
    state["messages"].append(output)
    state["answer"] = output
    return state

def router_agent(state):
    """Router agent that decides which specialized agent should handle the query."""
    state["messages"].append("Router received the query and will decide next agent.")
    return state

agent_docs = {
    "search_agent": search_agent.__doc__,
    "summarizer_agent": summarizer_agent.__doc__,
}

def routing_logic(state):
    prompt = f"""
    You are a router agent. Decide whether this query should go to 'summarizer_agent' or 'search_agent'. 
    Query: {state['messages'][0]}
    Respond with exactly one agent name:
    - "summarizer_agent"
    - "search_agent"
    
    Important: Tools only accept JSON arguments. Example:
    {{"name": "web_search", "arguments": {{"query": "example"}}}}
    """
    response = llm.invoke(prompt)
    decision = response.content.lower()
    if "summarizer" in decision:
        return "summarizer_agent"
    return "search_agent"