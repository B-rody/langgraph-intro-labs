# SEQUENTIAL AGENT PIPELINE
#
# This file demonstrates a manually orchestrated pipeline where we:
# 1. Call the planner agent to create a report outline
# 2. Extract the outline and pass it to the researcher agent
# 3. Extract the research notes and pass them to the writer agent
# 4. Extract the final report
#
# This is the "traditional" way to chain agents together - explicitly calling
# each one in sequence and manually managing the data flow between them.
#
# Key concepts:
# - Sequential execution: Each agent waits for the previous one to complete
# - Manual data extraction: We pull specific data from each response
# - Explicit handoffs: We create new messages for each agent using the previous output
#
# In Module 4, we'll see how LangGraph automates this orchestration using a graph.

import agents
from prompts import USER
from langchain.messages import HumanMessage

async def run_writer_pipeline(topic: str, audience: str):
    """
    Runs the full sequential pipeline: planning, researching, and writing.
    
    Pipeline stages:
    1. Planner: Creates a structured outline for the report
    2. Researcher: Gathers information for each section in the outline
    3. Writer: Drafts the full report using the research notes
    
    Args:
        topic: The subject of the report (e.g., "LangChain vs Microsoft Agent Framework")
        audience: Who the report is for (e.g., "Non-technical founders")
    
    Returns:
        The final report draft as a string
    """
    # Create the initial user message with the topic and audience
    # This message will be sent to the planner agent to start the pipeline
    user_message = HumanMessage(USER.REPORT_REQUEST_PROMPT.format(topic=topic, audience=audience))

    print("user_message:", user_message)

    # STAGE 1: PLANNER AGENT
    # Get the planner agent (this may initialize it if it doesn't exist yet)
    planner_agent = await agents.get_planner_agent()
    
    # Invoke the planner with the user's request
    # The planner will create a structured outline for the report
    # Note: We use .invoke() (synchronous) for the planner, .ainvoke() (async) for others
    # This is just showing both styles - in production, pick one approach consistently
    plan_response = planner_agent.invoke(
        {"messages": [user_message]}
    )

    # Extract the outline from the planner's response
    # plan_response['messages'][-1] gets the last message (the AI's response)
    # .content gets the actual text content of that message
    # For agents with structured output, this will be a JSON string representation
    report_outline = plan_response['messages'][-1].content
    
    # Print the structured outline for debugging/visibility
    # The structured_response is the validated Pydantic model (ReportSchema)
    # .model_dump_json() converts it to a pretty-printed JSON string
    print("********************REPORT OUTLINE********************")
    print(f"Structured Output:\n{plan_response['structured_response'].model_dump_json(indent=2)}\n")


    # STAGE 2: RESEARCHER AGENT
    # Get the researcher agent
    researcher_agent = await agents.get_researcher_agent()
    
    # Create a new message for the researcher containing the outline
    # This is the key "handoff" - we take the output from the planner and
    # create a fresh message for the researcher to work with
    # The researcher will use this outline to know what topics to research
    research_request_message = HumanMessage(report_outline)
    
    # Invoke the researcher with the outline
    # The researcher will gather information for each section in the outline
    # It will use tools (web search, MCP documentation) to find real sources
    research_response = await researcher_agent.ainvoke(
        {"messages": [research_request_message]}
    )

    # Extract the research notes from the researcher's response
    # This contains structured research findings with citations
    research_notes = research_response['messages'][-1].content
    
    # Print the structured research notes
    # The structured_response is the validated Pydantic model (ResearchNotesSchema)
    print("********************RESEARCH NOTES********************")
    print(f"Structured Output:\n{research_response['structured_response'].model_dump_json(indent=2)}\n")


    # STAGE 3: WRITER AGENT
    # Get the writer agent
    writer_agent = await agents.get_writer_agent()
    
    # Create a new message for the writer containing the research notes
    # This is the second "handoff" - we take the researcher's output and
    # create a fresh message for the writer to work with
    # The writer will use these notes to draft the full report
    writing_request_message = HumanMessage(research_notes)
    
    # Invoke the writer with the research notes
    # The writer will compose a complete report using the structured research
    writing_response = await writer_agent.ainvoke(
        {"messages": [writing_request_message]}
    )

    # Extract the final report from the writer's response
    # Note the different access pattern: [0]["text"]
    # This is because the writer agent has tools but no structured output,
    # so its content is formatted as a list of content blocks
    # [0] gets the first content block, ["text"] extracts the text from it
    #
    # Compare this to planner/researcher where we just used .content directly
    # because they have structured output (JSON strings)
    final_report = writing_response['messages'][-1].content[0]["text"]

    # Print the final report
    print("********************FINAL REPORT********************")
    print(final_report)
    
    # Return the final report so it can be used by the caller if needed
    return final_report