from pydantic import BaseModel, ConfigDict, Field
from typing import List, Literal

# Define the schema (structure) for a single section of the report
# Pydantic BaseModel lets us define exactly what data we expect and validate it automatically
# This ensures the AI returns data in a predictable format we can work with
class _SectionSchema(BaseModel):
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
class OutlineSchema(BaseModel):
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
    sections: List[_SectionSchema] = Field(
        ...,
        description=(
            "A list of major sections in the report. Each section contains a heading "
            "and a list of supporting bullet points. For a 1,000-word outline, "
            "there should typically be 5-7 sections."
        ),
        min_length=5,
        max_length=7
    )


# Define the schema for a single citation/source reference
# This represents one piece of evidence that supports a research note
class _Citation(BaseModel):
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
class _ResearchNote(BaseModel):
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
    citations: List[_Citation] = Field(
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
    notes: List[_ResearchNote] = Field(
        ...,
        description=(
            "A list of focused research notes covering the main aspects of the "
            "topic. Each note should be reusable when drafting the report."
        ),
        min_length=3,
        max_length=10
    )

# Define the schema for the critic agent's evaluation
# This represents the structured feedback a critic provides when reviewing a report draft
class CriticSchema(BaseModel):
    """
    Represents the structured output for the critic agent.
    """
    # Forbid extra fields to ensure clean, predictable output
    model_config = ConfigDict(extra="forbid")

    # score: a numerical rating from 1-10 indicating overall quality
    # 1 = poor quality, needs major revision
    # 10 = excellent, publication-ready
    # ge=1 and le=10 enforce the valid range (greater than or equal to 1, less than or equal to 10)
    score: int = Field(
        ...,
        description="An overall score for the report draft, from 1 (poor) to 10 (excellent).",
        ge=1,
        le=10
    )

    # feedback: list of detailed written critique explaining the score
    # Should provide actionable suggestions for improvement
    # This helps the writer understand what needs to be revised
    feedback: List[str] = Field(
        ...,
        description="Constructive feedback on the report draft, focusing on improvements."
    )