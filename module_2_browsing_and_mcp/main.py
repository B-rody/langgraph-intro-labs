# Import necessary components from LangChain and related libraries
# create_agent: creates an agent that can use the chat model
# init_chat_model: initializes a connection to a chat model (like GPT)
# ProviderStrategy: helps enforce structured output format
# InMemorySaver: stores conversation history in memory
# load_dotenv: loads environment variables from a .env file (including your API key)
# BaseModel, ConfigDict, Field: Pydantic classes for defining data schemas
# List, Literal: Python type hints for lists and restricted string values
# HumanMessage: represents a message from the user
# MultiServerMCPClient: client for connecting to MCP (Model Context Protocol) servers
#   MCP servers expose specialized tools (like documentation search) that agents can use
# asyncio: Python library for running asynchronous (non-blocking) code
#   Used here because MCP operations need async to avoid blocking while fetching data
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ProviderStrategy
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field
from typing import List
from langchain.messages import HumanMessage
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

# Define the schema for a single citation/source reference
# This represents one piece of evidence that supports a research note
class Citation(BaseModel):
    """
    Represents a single source reference used in the research process.
    """
    # ConfigDict(extra="forbid") means the AI cannot add extra fields beyond what we define
    # This prevents hallucinated or unexpected data from appearing in the output
    model_config = ConfigDict(extra="forbid")
    
    # url: the actual web address where this information came from
    # Must be a real URL returned by a tool call (browsing or MCP), never invented
    url: str = Field(
        ...,
        description="The direct URL to the source used for this note."
    )

    # title: a human-friendly name for this source (e.g., "LangChain Documentation - Agents")
    title: str = Field(
        ...,
        description="A short, human-readable title or label for the source."
    )

    # source_type: tracks whether this came from web browsing or the MCP documentation server
    # Literal["browsing", "mcp"] means only these two exact strings are allowed
    # This helps us understand which tool provided which information
    source_type: Literal["browsing", "mcp"] = Field(
        ...,
        description=(
            "Where this source came from."
        )
    )

    # snippet: a short quote or excerpt from the source that supports the research note
    # min_length=10 ensures we get meaningful content, not just a word or two
    snippet: str = Field(
        ...,
        description=(
            "A short excerpt or summary taken from the source that supports this "
            "note. A few sentences at most."
        ),
        min_length=10
    )


# Define the schema for a single research note
# This represents one key finding or insight from the research process
class ResearchNote(BaseModel):
    """
    Represents a single research note about one aspect of the topic or question.
    """
    # Again, forbid extra fields to prevent hallucinated data
    model_config = ConfigDict(extra="forbid")

    # heading: a short label for what this note is about (e.g., "Key Features", "Use Cases")
    heading: str = Field(
        ...,
        description=(
            "A short label for this note, such as a subtopic or section name. "
            "In later modules this can align with an outline section."
        )
    )

    # summary: the actual research finding - written clearly for the target audience
    # min_length=20 ensures we get substantial content, not just a sentence fragment
    # This summary should be detailed enough to be used directly in a report draft
    summary: str = Field(
        ...,
        description=(
            "A concise explanation of one key idea, finding, or point related to "
            "the research question. This should be written so it can be reused "
            "directly in a draft."
        ),
        min_length=20
    )

    # citations: a list of 1-5 Citation objects that back up this research note
    # Each citation proves that this information came from a real source
    # min_length=1 means every note must have at least one supporting source
    citations: List[Citation] = Field(
        ...,
        description=(
            "List of sources that support this note. Each citation must correspond "
            "to a real URL obtained from a tool call."
        ),
        min_length=1,
        max_length=5
    )


# Define the top-level schema for the complete research output
# This is what the agent will return after completing its research
class ResearchNotesSchema(BaseModel):
    """
    Structured output for the research agent.

    This object is designed to be consumed later by a drafting agent that turns
    notes and citations into a full report.
    """
    # Forbid extra fields to maintain clean, predictable structure
    model_config = ConfigDict(extra="forbid")
    
    # topic: the broad subject area being researched
    topic: str = Field(
        ...,
        description="High-level topic or subject of the research."
    )

    # question: the specific question the research is trying to answer
    question: str = Field(
        ...,
        description="The concrete research question or user request being answered."
    )

    # audience: who this research is for (affects the tone and depth of summaries)
    audience: str = Field(
        ...,
        description=(
            "Description of the intended audience "
            "(for example 'non-technical founders' or 'junior ML engineers')."
        )
    )

    # notes: a list of 3-10 ResearchNote objects covering different aspects of the topic
    # This nested structure creates: ResearchNotesSchema -> ResearchNote -> Citation
    # Each layer adds more detail and structure to the research findings
    notes: List[ResearchNote] = Field(
        ...,
        description=(
            "A list of focused research notes covering the main aspects of the "
            "topic. Each note should be reusable when drafting the report."
        ),
        min_length=3,
        max_length=10
    )


# Load environment variables from the .env file
# This reads your OPENAI_API_KEY so you don't have to hard-code it
load_dotenv()

# Configure the MCP (Model Context Protocol) client
# MCP is a protocol that lets AI agents connect to external data sources and tools
# In this case, we're connecting to LangChain's official documentation server
# The MCP server exposes tools for searching and retrieving documentation
mcp_client = MultiServerMCPClient(
    {
        # "langchain-mcp-server" is just a name we're giving this connection
        "langchain-mcp-server": {
            # "transport": how we communicate with the server (HTTP in this case)
            "transport": "streamable_http",
            # "url": the web address of LangChain's MCP documentation server
            "url": "https://docs.langchain.com/mcp"
        }
    }
)

# Get the list of tools (operations) the MCP server provides
# asyncio.run() executes the async function and waits for it to complete
# The MCP server defines tools like "search documentation", "get page content", etc.
# We don't need to define these tools ourselves - the server provides them
# These tools will be automatically available to our agent
research_agent_tools = asyncio.run(mcp_client.get_tools())

# System prompt: defines the AI's role as a research assistant and how to use tools
# This is more complex than Module 1 because the agent now has access to external tools
# The prompt teaches the agent:
#   - WHEN to use documentation tools vs. web browsing (MCP for official info, browsing for broader context)
#   - HOW to cite sources (only use real URLs from tool calls, never invent them)
#   - WHAT format to return (structured ResearchNotesSchema, no extra commentary)
# Key guideline: prefer authoritative documentation (MCP) over general web content when available
# This reduces hallucinations by grounding the agent's answers in official sources
system_prompt = """
You are an expert research assistant. Your task is to gather accurate information using the
tools available to you, then summarize your findings clearly and concisely in structured form.

Behavior Guidelines:
1. Use the tools available to you to retrieve real information. These tools may include
   web browsing, document retrieval, or other external-source operations.
2. When a tool can provide authoritative definitions, API details, or official explanations,
   prefer those sources over open-web content.
3. If you are going to look up information related to LangChain, prioritize using the MCP tool
   to access official LangChain documentation and resources.
4. When broader context, examples, tutorials, or recent discussions are needed, use tools
   that can search or retrieve open-web content.
5. For every citation, include only URLs that were actually returned by a tool call.
   Never invent, guess, or approximate URLs.
6. For each major point, create a research note with:
   - a short heading,
   - a clear multi-sentence summary written for the specified audience,
   - at least one supporting citation.
7. Produce only the structured output defined by the ResearchNotesSchema. Do not include
   commentary, markdown fences, or descriptions of your reasoning process.
8. If tool results are ambiguous or incomplete, make additional tool calls as needed
   before forming your final notes.

Your goal is to produce high-quality, tool-verified research that can be used directly
in downstream writing and revision tasks.
""" 

# User message template: the specific research task for the agent
# Like Module 1, we use placeholders {question}, {topic}, {audience} for reusability
# This template tells the agent:
#   - What question to answer
#   - Who the audience is (affects tone and depth)
#   - How to structure the output (ResearchNotesSchema)
#   - Important: only use real URLs from tool calls, never make them up
user_message_text = """ 
You are a research assistant. Please gather accurate, tool-verified information and return it
using the ResearchNotesSchema.

Research Question: {question}
Topic: {topic}
Audience: {audience}

Requirements:l
1. Use the browsing and MCP tools as needed to gather real information.
2. Create multiple research notes (3-10), each with:
   - a heading describing the specific subtopic or aspect,
   - a clear summary written for the specified audience,
   - at least one citation with a real URL retrieved by a tool.
3. All citations must come from actual tool calls. Do not fabricate or guess URLs.
4. Do not include any commentary, markdown, or extra text. Return only structured content.
"""

# Initialize the chat model with low temperature for consistency
# temperature=0.2 keeps research output predictable and factual
model = init_chat_model(
    "gpt-5-mini", 
    temperature=0.2
)

# Add the provider's built-in web search/browsing capability to our tools
# {"type": "web_search"} tells LangChain to enable the model provider's browsing feature
# This lets the agent search the web and read web pages without us building a custom search tool
# Now the agent has two types of tools:
#   1. MCP tools (from research_agent_tools) - for LangChain documentation
#   2. Web search - for broader internet content
research_agent_tools.append({"type": "web_search"})

# Create the research agent with all the components:
# - model: the AI that will do the thinking
# - system_prompt: instructions on how to behave and use tools
# - tools: both MCP documentation tools AND web browsing
# - response_format: forces output to match ResearchNotesSchema
#
# .with_retry() adds automatic retry logic:
# - stop_after_attempt=3: try up to 3 times if the request fails
# - exponential_jitter_params: controls retry timing
#   * initial=0.2: wait 0.2 seconds before first retry
#   * max=4: never wait more than 4 seconds between retries
#   * exp_base=2: double the wait time with each retry (0.2s, 0.4s, 0.8s...)
#   * jitter=0.1: add random variation to avoid retry storms
# Retries help when:
#   - The model fails to produce valid structured output
#   - API calls timeout or have temporary errors
#   - Tool calls fail temporarily
research_agent = create_agent(
    model,
    system_prompt=system_prompt,
    tools=research_agent_tools,
    response_format=ProviderStrategy(ResearchNotesSchema)
).with_retry(
    stop_after_attempt=3,
    exponential_jitter_params={
        "initial": 0.2,
        "max": 4,
        "exp_base": 2,
        "jitter": 0.1
    }
)

# Define the specific research task parameters
# These values will be inserted into the user_message_text template
topic = "LangChain"
audience = "Beginner AI developers"
question = "How does LangChain compare with Microsoft's Agent Framework?"

# Create the actual user message by filling in the template placeholders
# .format() replaces {topic}, {audience}, and {question} with the actual values
# HumanMessage() wraps it in the format LangChain expects
user_message = HumanMessage(
    user_message_text.format(
        topic=topic,
        audience=audience,
        question=question
    )
)

# Invoke the agent asynchronously to perform the research
# asyncio.run() executes the async function and waits for it to complete
# We use .ainvoke() (async invoke) instead of .invoke() because:
#   - The agent needs to call MCP tools which are async
#   - Tool calls might take time (web searches, documentation lookups)
#   - Async allows efficient handling of multiple tool calls without blocking
# The agent will:
#   1. Analyze the question
#   2. Decide which tools to use (MCP for LangChain docs, browsing for Microsoft info)
#   3. Make multiple tool calls to gather information
#   4. Synthesize the information into ResearchNotesSchema format
#   5. Return structured notes with real citations from tool calls
result = asyncio.run(research_agent.ainvoke(
    {"messages": [user_message]}
))


# Print the results in different formats
print("********************RESEARCH AGENT RESPONSE********************")
# The complete result object (includes all metadata and tool call logs)
print(f"AI Response: {result}\n")
# Just the text content of the AI's response
print(f"AI Response Content: {result['messages'][-1].content}\n")
# The structured output as JSON - this is the validated ResearchNotesSchema object
# This contains:
#   - topic, question, audience
#   - a list of research notes, each with:
#     * heading (subtopic)
#     * summary (the actual research finding)
#     * citations (real URLs from tool calls with snippets)
# model_dump_json() converts the Pydantic model to pretty-printed JSON
print(f"Structured Output:\n{result['structured_response'].model_dump_json(indent=2)}\n")