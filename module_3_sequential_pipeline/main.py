# MODULE 3: Sequential Agent Pipeline
#
# This module demonstrates a manually orchestrated, sequential pipeline of agents.
# We explicitly control the flow: planner → researcher → writer
#
# Key characteristics of a sequential pipeline:
# - Manual orchestration: You write code that calls each agent in order
# - Explicit data passing: You extract output from one agent and pass it to the next
# - Procedural: Reads like a step-by-step recipe
# - Simple but rigid: Easy to understand, but hard to add branching or parallel work
#
# The actual pipeline logic is in pipeline.py - see that file for detailed comments
# on how each stage works and how data flows between agents.
#
# In Module 4, we'll convert this same workflow into LangGraph format,
# which provides more flexibility for complex workflows.

import pipeline
import asyncio

# Run the sequential pipeline with a specific topic and audience
# This will execute: planner → researcher → writer → output
asyncio.run(
    pipeline.run_writer_pipeline(
        topic="LangChain versus Microsoft Agent Framework",
        audience="Non-technical founders"
    )
) 