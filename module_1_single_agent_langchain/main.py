# Import necessary components from LangChain and related libraries
# create_agent: creates an agent that can use the chat model
# init_chat_model: initializes a connection to a chat model (like GPT)
# ProviderStrategy: helps enforce structured output format (makes AI return data in a specific schema)
# InMemorySaver: stores conversation history in memory
# load_dotenv: loads environment variables from a .env file (including your API key)
# BaseModel, Field: Pydantic classes for defining data schemas (structure of the output)
# List: Python type hint for lists
# HumanMessage: represents a message from the user
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.structured_output import ProviderStrategy
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
from langchain.messages import HumanMessage


# Define the schema (structure) for a single section of the report
# Pydantic BaseModel lets us define exactly what data we expect and validate it automatically
# This ensures the AI returns data in a predictable format we can work with
class SectionSchema(BaseModel):
    """
    Represents one section of the report outline.
    """

    # heading: the title of this section (must be a string)
    # The "..." means this field is required (cannot be empty)
    heading: str = Field(
        ...,
        description="The title of the section. Must be a non-empty string."
    )

    # bullets: a list of 2-3 bullet points for this section
    # min_length and max_length enforce that we get exactly 2-3 bullets
    # List[str] means "a list where each item is a string"
    bullets: List[str] = Field(
        ...,
        description=(
            "A list of short, concise bullet points explaining the section's "
            "main idea. Typically 2-3 bullets. Each bullet should be a "
            "non-empty string."
        ),
        min_length=2,
        max_length=3
    )


# Define the schema for the complete report outline
# This is the top-level structure that contains the title and all sections
class ReportSchema(BaseModel):
    """
    Represents the full outline for a 1,000-word report.
    """

    # title: the overall title of the report
    title: str = Field(
        ...,
        description="The overall title of the report. Must be a non-empty string."
    )

    # sections: a list of 5-7 SectionSchema objects (each section we defined above)
    # By using List[SectionSchema], we're saying "this is a list where each item follows the SectionSchema structure"
    # This creates a nested structure: Report contains Sections, each Section contains a heading and bullets
    sections: List[SectionSchema] = Field(
        ...,
        description=(
            "A list of major sections in the report. Each section contains a heading "
            "and a list of supporting bullet points. For a 1,000-word outline, "
            "there should typically be 5-7 sections."
        ),
        min_length=5,
        max_length=7
    )

# Load environment variables from the .env file
# This reads your OPENAI_API_KEY so you don't have to hard-code it
load_dotenv()

# System prompt: defines the AI's role and behavior guidelines
# This is like giving the AI its "job description" and "company policies"
# The system prompt persists across all messages and shapes how the AI responds
# Here, we're telling it to be a professional report planner that follows instructions precisely
system_prompt = """ 
You are an expert report planner. Your task is to generate structured report outlines that follow the user's requirements exactly. 
Behavior guidelines: 
1. Always stay professional, clear, and concise. 
2. Follow every formatting rule provided by the user prompt. 
3. Never include explanations, reasoning steps, or commentary about your process. 
4. If the user asks for a structured or JSON format, output only that structure—no extra text, markdown fences, or code blocks. 
5. Maintain consistent formatting and section numbering between runs. 
Your goal is to deliver a complete, polished outline that is ready for downstream use without manual cleanup. 
""" 

# User message template: this is the actual task we'll send to the AI
# Notice the {topic} and {audience} placeholders - we'll fill these in later with real values
# This template approach lets us reuse the same prompt structure for different topics/audiences
# The detailed requirements help the AI produce consistent, predictable output
user_message_text = """ 
Produce a clean, structured outline for a ~1,000-word report. 
Topic: {topic} 
Audience: {audience} 
Requirements: 
1. Start with a title line prefixed with "#". 
2. Include 5-7 numbered sections, each prefixed with "##" followed by the section number and title. 
3. Under each section, include 2-3 short points, each starting with a dash. 
4. Do not include commentary, markdown fences, or code blocks. 
5. Keep the language concise and appropriate for the audience. 
Return only the formatted outline. 
"""

# Initialize the chat model with specific parameters
# temperature=0.2 makes the output more consistent and predictable
# Temperature range: 0.0 (very deterministic, same input similar output) to 2.0 (very creative, random)
# Low temperature (0.2) is good for structured tasks where consistency matters
model = init_chat_model(
    "gpt-5-nano", 
    temperature=0.2
)

# Create the agent that will generate report outlines
# This combines the model with additional configuration:
# - system_prompt: gives the agent its role and behavior rules
# - response_format: forces the AI to return data matching our ReportSchema
#   ProviderStrategy tells LangChain to use the model provider's native structured output feature
# - checkpointer: enables conversation memory (though we're only doing one request here)
planner_agent = create_agent(
    model,
    system_prompt=system_prompt,
    response_format=ProviderStrategy(ReportSchema)
)

# Define the specific topic and audience for this report
# These values will be inserted into the user_message_text template
topic = "Responsible AI"
audience = "non-technical founders"

# Create the actual user message by filling in the template placeholders
# .format() replaces {topic} and {audience} with the actual values
# HumanMessage() wraps it in the format LangChain expects
user_message = HumanMessage(
    user_message_text.format(
        topic=topic,
        audience=audience
    )
)

# Invoke the agent with our message
# This sends the request to the AI and waits for the structured response
result = planner_agent.invoke(
    {"messages": [user_message]}
)

# Print the results in different formats
print("********************REPORT OUTLINE RESPONSE********************")
# The complete result object (includes all metadata)
print(f"AI Response: {result}\n")
# Just the text content of the AI's response
print(f"AI Response Content: {result['messages'][-1].content}\n")
# The structured output as JSON - this is the validated ReportSchema object
# model_dump_json() converts the Pydantic model to pretty-printed JSON
# This structured data is easy to use in code (access with result['structured_response'].title, etc.)
print(f"Structured Output:\n{result['structured_response'].model_dump_json(indent=2)}\n")
