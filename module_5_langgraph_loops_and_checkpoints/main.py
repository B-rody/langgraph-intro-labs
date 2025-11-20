# MODULE 5: LangGraph Loops and Conditional Routing
#
# In Module 4, we built a linear graph: planner → researcher → writer → END
# The workflow always followed the same path with no branches or loops.
#
# In Module 5, we add:
# - LOOPS: The writer can revise the draft multiple times based on feedback
# - CONDITIONAL EDGES: Graph decides whether to continue revising or end based on critique score
# - ITERATIVE IMPROVEMENT: A critic agent evaluates drafts and provides feedback
#
# New workflow:
#   planner → researcher → writer → critic → (loop back to writer OR end)
#
# Key concepts:
# - Conditional edges: Functions that return node names to route to next
# - Cycles: Edges that loop back to previous nodes for revision
# - State tracking: critique_count and critiques list track iteration progress
# - Exit conditions: Stop when score ≥ 8 OR after 3 critique rounds
#
# This demonstrates how LangGraph enables complex workflows with:
# - Feedback loops for iterative refinement
# - Dynamic routing based on agent outputs
# - Built-in retry policies for error handling

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
from langgraph.types import RetryPolicy


# Define the shared state for the graph
# Compared to Module 4, we've added two new fields for tracking critique iterations:
# - critiques: A list of critique results from each revision round
# - critique_count: Tracks how many critique rounds have been completed
#
# ReportState contains:
# - messages: Full conversation history (Annotated with operator.add means messages accumulate)
# - outline: The structured report outline from the planner (stored as a dict)
# - research_notes: The research findings from the researcher (stored as a dict)
# - draft: The current report text from the writer (updated with each revision)
# - critiques: List of dicts with {score: int, feedback: List[str]} from each critique round
#   * Annotated with operator.add so critiques accumulate across iterations
# - critique_count: Integer counter tracking which revision round we're on (0, 1, 2, 3...)
#
# Each node function can:
# - READ from state: access outline, research_notes, draft, critiques, etc.
# - WRITE to state: return a dict with updated values
# LangGraph automatically merges the returned dict into the state
class ReportState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    outline: dict
    research_notes: dict
    draft: str
    critiques: Annotated[List[dict], operator.add] # List of critique results. Annotated with operator.add to accumulate critiques across revision rounds
    critique_count: int # Track which revision round we're on (0, 1, 2...)


# Initialize all four agents in parallel at startup
# Module 4 had three agents (planner, researcher, writer)
# Module 5 adds a fourth: the critic agent to evaluate drafts
#
# The critic_agent uses CriticSchema to return structured feedback:
# - score: int (1-10) rating the draft quality
# - feedback: List[str] of specific improvement suggestions
#
# This async function initializes all agents concurrently using asyncio.gather()
# to reduce startup time compared to sequential initialization
async def init_agents():
    return await asyncio.gather(
        agents.get_planner_agent(),
        agents.get_researcher_agent(),
        agents.get_writer_agent(),
        agents.get_critic_agent()
    )

planner_agent, researcher_agent, writer_agent, critic_agent = asyncio.run(init_agents())

# Helper function to extract text content from AI messages
# Many agents return AIMessage with complex content structures:
# - Agents with structured output: msg.content is a simple string
# - Agents with tools (like writer): msg.content is a list of content blocks
#
# This function handles both formats:
# 1. If content is a string, return it directly
# 2. If content is a list, find the text block and extract it
#
# This is used in the writer node to extract the draft text from tool-using agent responses
# and in routing functions to check if meaningful content exists before routing to the next node
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
# In Module 4: This node ran once to create the final report
# In Module 5: This node can run multiple times, incorporating critic feedback
#
# This node:
# 1. On first run (critique_count == 0):
#    - Reads research_notes from state
#    - Creates initial draft from research
# 2. On subsequent runs (critique_count > 0):
#    - Reads the latest critique feedback
#    - Revises the draft based on specific suggestions
# 3. Returns updated state with:
#    - messages: The conversation with the writer
#    - draft: The new/revised report text (extracted using get_last_ai_text helper)
#
# The writer agent doesn't return structured output (it has tools enabled),
# so we use get_last_ai_text() to extract the text from content blocks
async def writer_agent_func(state: ReportState):
    notes = state["research_notes"]
    
    # First draft: use research notes
    if state["critique_count"] == 0:
        prompt = f"""
Write a report using only the following research notes:
{json.dumps(notes, indent=2)}
    """
    # Revision: incorporate critic feedback
    else:
        latest_critique = state["critiques"][-1] if state["critiques"] else {}
        feedback = latest_critique.get("feedback", [])
        current_draft = state["draft"]
        
        prompt = f"""
You are revising a report based on critic feedback.

CURRENT DRAFT:
{current_draft}

CRITIC FEEDBACK:
{feedback}

Please revise the draft to address the feedback while keeping the research foundation.
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

# NODE 4: Critic Agent (NEW in Module 5)
# This node enables iterative improvement of the report draft
# Module 4 didn't have a critique/revision loop - it just went planner → researcher → writer → END
# Module 5 adds this critic to evaluate drafts and provide actionable feedback
#
# This node:
# 1. Reads the current draft from state
# 2. Calls the critic agent to evaluate quality (returns CriticSchema with score + feedback)
# 3. Returns updated state with:
#    - messages: The conversation with the critic
#    - critiques: A list element with {score: int, feedback: List[str]}
#      * Uses Annotated[operator.add] so each critique appends to the list
#    - critique_count: Incremented by 1 to track revision rounds
#
# The routing functions (route_from_writer, route_from_critic) use:
# - critique_count to enforce max 3 revision rounds
# - score to decide if quality is acceptable (≥8 means done)
async def critic_agent_func(state: ReportState):
    draft = state["draft"]
    prompt = f"""
Please critique the following report draft:
{draft}
    """
    critique_request_message = HumanMessage(prompt)
    critique_response = await critic_agent.ainvoke(
        {"messages": [critique_request_message]}
    )

    critique_count = state.get("critique_count", 0) + 1
    critique_results = [{
        "score": critique_response["structured_response"].score,
        "feedback": critique_response["structured_response"].feedback,
    }]

    return {
        "messages": critique_response["messages"],
        "critiques": critique_results,
        "critique_count": critique_count
    }

# ROUTING FUNCTION: Decide where to go after writer_agent
# Module 4 used unconditional edges: planner → researcher → writer → END
# Module 5 uses CONDITIONAL edges that make routing decisions based on state
#
# This routing function runs after writer_agent completes and decides:
# - Send to "critic_agent" if: critique_count < 3 AND (no critiques yet OR last score < 8)
#   * We haven't reached max revision rounds (3)
#   * Quality isn't good enough yet (score < 8 means needs improvement)
# - Send to END if: we've done 3 rounds OR last score ≥ 8
#   * Either max revisions reached or quality is acceptable
#
# Return value MUST be a string matching a node name or END
# LangGraph uses this string to route to the next node
async def route_from_writer(state: ReportState) -> str:
    """
    Determines whether to send draft to critic for evaluation
    or finish the workflow based on critique count and quality.
    """
    # If we haven't done 3 critiques yet and we've either not done any critiques or the last score is below 8, loop back to critic
    if state["critique_count"] < 3 and (len(state["critiques"]) == 0 or state["critiques"][-1]["score"] < 8):
        return "critic_agent"  # Send to critic for evaluation
    return END  # Quality acceptable or max revisions reached

# ROUTING FUNCTION: Decide where to go after critic_agent
# This routing function runs after critic_agent evaluates a draft and decides:
# - Send to "writer_agent" if: score < 8
#   * Quality needs improvement, writer should revise based on feedback
# - Send to END if: score ≥ 8
#   * Quality is acceptable, workflow complete
#
# This creates a loop: writer → critic → writer → critic → ... until done
# The loop terminates when either:
# 1. Score reaches 8+ (quality threshold met)
# 2. route_from_writer hits the 3-round limit
async def route_from_critic(state: ReportState) -> str:
    """
    Determines whether to loop back to the writer for revisions
    or finish based on the critic's quality score.
    """
    if state["critiques"][-1]["score"] < 8:
        return "writer_agent"  # Loop back to writer for revision
    return END # Quality acceptable, finish workflow

# BUILD THE GRAPH
# In Module 4: We built a graph with 3 nodes and linear flow: planner → researcher → writer → END
# In Module 5: We have 4 nodes with conditional routing and loops for iterative improvement
#
# StateGraph creates a workflow where:
# - Nodes are functions that process/transform state
# - Edges connect nodes (unconditional) or route dynamically (conditional)
# - START is the entry point, END is the exit point
# - State flows through the graph and accumulates data
graph_builder = StateGraph(ReportState)

# Add four nodes to the graph (Module 4 had 3, Module 5 adds critic)
# Each node is associated with a function that will be called when that node executes
# RetryPolicy(retry_on=[ValueError]) means if the function raises ValueError, retry automatically
graph_builder.add_node("planner_agent", planner_agent_func, retry_policy=RetryPolicy(retry_on=[ValueError]))
graph_builder.add_node("researcher_agent", researcher_agent_func, retry_policy=RetryPolicy(retry_on=[ValueError]))
graph_builder.add_node("writer_agent", writer_agent_func, retry_policy=RetryPolicy(retry_on=[ValueError]))
graph_builder.add_node("critic_agent", critic_agent_func, retry_policy=RetryPolicy(retry_on=[ValueError]))

# UNCONDITIONAL EDGES: These always route to the same next node
# Module 4 only had unconditional edges (simple linear flow)
# Module 5 combines unconditional edges (for the initial pipeline) with conditional edges (for the loop)
graph_builder.add_edge(START, "planner_agent")  # Always start with planner
graph_builder.add_edge("planner_agent", "researcher_agent")  # Planner always goes to researcher
graph_builder.add_edge("researcher_agent", "writer_agent")  # Researcher always goes to writer

# CONDITIONAL EDGES: These call a routing function to decide the next node
# This is NEW in Module 5 - Module 4 didn't have any conditional routing
#
# add_conditional_edges syntax:
# - First arg: source node name (where the edge starts)
# - Second arg: routing function that returns a string (node name or END)
# - Third arg: path dictionary mapping return values to actual destinations
#
# Example: After writer_agent completes, route_from_writer() is called
# - If it returns "critic_agent", LangGraph routes to critic_agent node
# - If it returns END, LangGraph terminates the workflow
graph_builder.add_conditional_edges(
    "writer_agent",
    route_from_writer,
    {
        "critic_agent": "critic_agent",  # Route to critic if draft needs evaluation
        END: END  # Or finish if quality is good or max rounds reached
    }
)
graph_builder.add_conditional_edges(
    "critic_agent",
    route_from_critic,
    {
        "writer_agent": "writer_agent",  # Loop back to writer if score < 8
        END: END  # Or finish if score ≥ 8
    }
)

# Compile the graph into an executable workflow
# After compilation, we can invoke the graph with an initial state
# The graph will follow the edges (unconditional and conditional) to execute the workflow
graph = graph_builder.compile()


# EXECUTE THE GRAPH
# In Module 4: We invoked the graph with 4 fields (messages, outline, research_notes, draft)
# In Module 5: We add 2 new fields to track critique iterations: critique_count and critiques
#
# The initial state provides:
# - messages: Empty list (will be populated by agents as they run)
# - outline, research_notes, draft: None (will be filled in by respective nodes)
# - critique_count: 0 (increments with each critique round)
# - critiques: [] (accumulates score/feedback from each critique)
#
# LangGraph execution flow:
# 1. Start at START → planner_agent
# 2. planner_agent → researcher_agent (unconditional edge)
# 3. researcher_agent → writer_agent (unconditional edge)
# 4. writer_agent → route_from_writer decides:
#    a. critic_agent (if count < 3 AND no critiques yet OR last score < 8)
#    b. END (if quality acceptable or max rounds reached)
# 5. IF critic_agent runs → route_from_critic decides:
#    a. writer_agent (if score < 8, loop back for revision)
#    b. END (if score ≥ 8, quality acceptable)
# 6. This loop continues until: score ≥ 8 OR critique_count ≥ 3
try: 
    result = asyncio.run(graph.ainvoke({
        "messages": [],
        "outline": None,
        "research_notes": None,
        "draft": None,
        "critique_count": 0,
        "critiques": []
}))
except Exception as e:
    print(f"Error during graph execution: {e}")
    import traceback
    traceback.print_exc()
    raise

# Save the complete state to a JSON file for debugging
# Module 4 saved: messages, outline, research_notes, draft
# Module 5 also saves: critiques and critique_count
#
# This helps debug the revision loop by showing:
# - How many critique rounds occurred
# - What scores and feedback were given each round
# - How the draft evolved across revisions
serializable = {
    "messages": [
        message_to_dict(m) if isinstance(m, BaseMessage) else m
        for m in result["messages"]
    ],
    "outline": result.get("outline"),
    "research_notes": result.get("research_notes"),
    "draft": result.get("draft"),
    "critiques": result.get("critiques", []),
    "critique_count": result.get("critique_count", 0)
}

with open("dump.json", "w", encoding="utf-8") as f:
    json.dump(serializable, f, indent=2, ensure_ascii=False)

# Print the final results
# Module 4 printed just the final draft
# Module 5 also prints critique history to show the iterative improvement process
print("********************FINAL REPORT********************")
final_report = result["draft"]
print(final_report)

# Print critique history (NEW in Module 5)
# Shows the iterative improvement: score and feedback from each revision round
if "critiques" in result and result["critiques"]:
    print("\n" * 3)
    print("********************CRITIQUE HISTORY********************")
    for idx, critique in enumerate(result["critiques"], start=1):
        print(f"\nCritique Round {idx}:")
        print(f"Score: {critique['score']}")
        print(f"Feedback: {critique['feedback']}\n")