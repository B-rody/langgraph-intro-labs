# MODULE 4: Transitioning from Sequential Pipeline to LangGraph
#
# In Module 3, we manually orchestrated a pipeline: planner → researcher → writer
# We explicitly called each agent, extracted outputs, and passed them to the next agent.
# This works but is rigid - hard to add conditional logic, loops, or parallel processing.
#
# In Module 4, we use LangGraph to define the same workflow as a graph:
# - Nodes: Individual agent functions (planner, researcher, writer)
# - Edges: Define the flow between nodes (who runs after whom)
# - State: A shared dictionary that all nodes can read from and write to
#
# Benefits of LangGraph:
# - Declarative: Define WHAT the workflow is, not HOW to execute it
# - Extensible: Easy to add conditional branching, cycles, or parallel nodes
# - Automatic: LangGraph handles execution, state management, and error recovery

import asyncio
import json
import operator
from typing import List

from typing_extensions import Annotated, TypedDict

from langchain.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.messages import BaseMessage, message_to_dict
from langgraph.graph import END, START, StateGraph

import agents
from prompts import USER


# Define the shared state for the graph
# In Module 3, we passed data between agents using variables (outline, research_notes, etc.)
# In LangGraph, we use a TypedDict called "State" that all nodes can access
#
# ReportState contains:
# - messages: Full conversation history (Annotated with operator.add means messages accumulate)
# - outline: The structured report outline from the planner (stored as a dict)
# - research_notes: The research findings from the researcher (stored as a dict)
# - draft: The final report text from the writer
#
# Each node function can:
# - READ from state: access outline, research_notes, etc.
# - WRITE to state: return a dict with updated values
# LangGraph automatically merges the returned dict into the state
class ReportState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    outline: dict
    research_notes: dict
    draft: str


# Initialize all three agents once at startup
# We use asyncio.gather() to initialize them in parallel (faster than sequential)
# This runs once when the module loads, not every time the graph executes
async def init_agents():
    return await asyncio.gather(
        agents.get_planner_agent(),
        agents.get_researcher_agent(),
        agents.get_writer_agent(),
    )

planner_agent, researcher_agent, writer_agent = asyncio.run(init_agents())

# Helper function to extract text content from AI messages
# The writer agent uses tools, so its response format is different:
# - Agents with structured output: msg.content is a simple string
# - Agents with tools (like writer): msg.content is a list of content blocks
#
# This function handles both formats:
# 1. If content is a string, return it directly
# 2. If content is a list, find the text block and extract it
#
# This is needed because Module 4's writer has tools but no structured output,
# unlike the planner and researcher which return structured data
def get_last_ai_text(state: ReportState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            # If it's just a string, we're good
            if isinstance(msg.content, str):
                if msg.content.strip():
                    return msg.content

            # If it's a list of content blocks (tool-using agents), find the text block
            if isinstance(msg.content, list):
                for block in reversed(msg.content):
                    if block.get("type") == "text" and block.get("text", "").strip():
                        return block["text"]

    raise ValueError("No AI text content found in state")

# NODE 1: Planner Agent
# In Module 3: We called planner_agent.invoke() directly in the pipeline function
# In Module 4: This is a graph node that LangGraph calls automatically
#
# Node functions in LangGraph:
# - Receive the current state as input
# - Perform work (call an agent, process data, etc.)
# - Return a dict with state updates
#
# This node:
# 1. Creates a message asking for a report outline on a specific topic
# 2. Calls the planner agent to generate the outline
# 3. Returns updated state with:
#    - messages: The conversation with the planner
#    - outline: The structured outline (converted from Pydantic model to dict)
#
# Note: We use .model_dump() to convert the Pydantic model to a plain dict
# This keeps our state simple and JSON-serializable
async def planner_agent_func(state: ReportState):
    topic = "LangChain versus Microsoft Agent Framework"
    audience = "Non-technical founders"
    planner_request_message = HumanMessage(USER.REPORT_REQUEST_PROMPT.format(topic=topic, audience=audience))
    planner_response = await planner_agent.ainvoke(
        {
            "messages": [planner_request_message]
        }
    )
    return {
        "messages": planner_response['messages'],
        "outline": planner_response['structured_response'].model_dump()
    }
    
# NODE 2: Researcher Agent
# In Module 3: We extracted the outline, formatted it, and called researcher_agent.ainvoke()
# In Module 4: LangGraph calls this node after the planner node completes
#
# This node:
# 1. Reads the outline from state (automatically available from planner's output)
# 2. Formats the outline as JSON for the researcher to understand
# 3. Calls the researcher agent to gather information about each section
# 4. Returns updated state with:
#    - messages: The conversation with the researcher
#    - research_notes: The structured research findings (converted to dict)
#
# Key difference from Module 3:
# - Module 3: We manually created a new message with the outline
# - Module 4: State automatically carries data between nodes; we just read state["outline"]
async def researcher_agent_func(state: ReportState):
    outline = state["outline"]
    prompt = f"""
Here is the structured outline:
{json.dumps(outline, indent=2)}

Please generate the research notes...
    """
    research_request_message = HumanMessage(prompt)
    research_response = await researcher_agent.ainvoke(
        {
            "messages": [research_request_message]
        }
    )
    return {
        "messages": research_response['messages'],
        "research_notes": research_response['structured_response'].model_dump()
    }

# NODE 3: Writer Agent
# In Module 3: We extracted research notes, formatted them, and called writer_agent.ainvoke()
# In Module 4: LangGraph calls this node after the researcher node completes
#
# This node:
# 1. Reads the research_notes from state (automatically available from researcher's output)
# 2. Formats the notes as JSON for the writer to use
# 3. Calls the writer agent to create a full report
# 4. Returns updated state with:
#    - messages: The conversation with the writer
#    - draft: The final report text (extracted using get_last_ai_text helper)
#
# Note: Unlike planner/researcher, the writer doesn't return structured output
# It has tools enabled, so we use get_last_ai_text() to extract the text from content blocks
async def writer_agent_func(state: ReportState):
    notes = state["research_notes"]

    prompt = f"""
Write a report using only the following research notes:
{json.dumps(notes, indent=2)}
    """

    writer_request_message = HumanMessage(prompt)
    writing_response = await writer_agent.ainvoke(
        {"messages": [writer_request_message]}
    )
    draft_text = get_last_ai_text({"messages": writing_response["messages"]})
    return {
        "messages": writing_response["messages"],
        "draft": draft_text,
    }
    


# BUILD THE GRAPH
# In Module 3: We manually called each agent in sequence using a function
# In Module 4: We define the workflow as a graph and let LangGraph execute it
#
# Graph building steps:
# 1. Create a StateGraph with our ReportState type
# 2. Add nodes: Each node is a function that processes the state
# 3. Add edges: Define the flow between nodes (who runs after whom)
# 4. Compile: Convert the graph definition into an executable workflow
#
# The graph structure:
#   START → planner_agent → researcher_agent → writer_agent → END
#
# This is equivalent to Module 3's sequential pipeline, but more flexible:
# - Easy to add conditional edges (if X then Y, else Z)
# - Easy to add parallel nodes (run multiple agents at once)
# - Easy to add cycles (feedback loops, retries, human-in-the-loop)
graph_builder = StateGraph(ReportState)

# Add three nodes to the graph
# Each node is associated with a function that will be called when that node executes
graph_builder.add_node("planner_agent", planner_agent_func)
graph_builder.add_node("researcher_agent", researcher_agent_func)
graph_builder.add_node("writer_agent", writer_agent_func)

# Add edges to define the execution flow
# START is a special node that represents the beginning of the graph
# END is a special node that represents the completion of the graph
graph_builder.add_edge(START, "planner_agent")
graph_builder.add_edge("planner_agent", "researcher_agent")
graph_builder.add_edge("researcher_agent", "writer_agent")
graph_builder.add_edge("writer_agent", END)

# Compile the graph into an executable workflow
# After compilation, we can invoke the graph with an initial state
graph = graph_builder.compile()


# EXECUTE THE GRAPH
# In Module 3: We called run_writer_pipeline(topic, audience)
# In Module 4: We invoke the graph with an initial state
#
# Key differences:
# - Module 3: Function controls execution step-by-step
# - Module 4: Graph controls execution based on edges
#
# The initial state provides:
# - messages: Empty list (will be populated by agents)
# - outline, research_notes, draft: None (will be filled in by nodes)
#
# LangGraph will:
# 1. Start at the START node
# 2. Follow edges: planner → researcher → writer → END
# 3. Each node updates the state
# 4. Return the final state when END is reached
try: 
    result = asyncio.run(graph.ainvoke({
        "messages": [],
        "outline": None,
        "research_notes": None,
        "draft": None,
}))
except Exception as e:
    print(f"Error during graph execution: {e}")
    import traceback
    traceback.print_exc()
    raise

# Save the message history to a JSON file for debugging
# This is useful to see exactly what each agent said during execution
# We convert BaseMessage objects to dicts so they can be serialized to JSON
serializable = {
    "messages": [
        message_to_dict(m) if isinstance(m, BaseMessage) else m
        for m in result["messages"]
    ]
}

with open("dump.json", "w", encoding="utf-8") as f:
    json.dump(serializable, f, indent=2, ensure_ascii=False)

# Print the final results
print("********************FINAL REPORT********************")
# Extract the final report from the state
# This was populated by the writer_agent node
final_report = result["draft"]
print(final_report)