from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.agents.structured_output import ProviderStrategy
import agent_mcp, prompts, agent_tools, schemas

async def get_planner_agent():
    """
    Initializes and returns a planner agent with memory capabilities.
    """
    load_dotenv()
    model = init_chat_model(
        "gpt-5-nano",
        temperature=0.2
    )
    agent = create_agent(
        model,
        system_prompt=prompts.PLANNER.SYSTEM_PROMPT,
        response_format=ProviderStrategy(schema=schemas.OutlineSchema)
    ).with_retry(
        stop_after_attempt=3,
        exponential_jitter_params={
            "initial": 0.2,
            "max": 4,
            "exp_base": 2,
            "jitter": 0.1
        }
    )
    return agent


async def get_researcher_agent():
    """
    Initializes and returns a researcher agent with memory capabilities.
    """
    load_dotenv()
    model = init_chat_model(
        "gpt-5-mini",
        temperature=0.2
    )
    researcher_tools = [ 
        *await agent_mcp.get_mcp_tools(),
        agent_tools.get_web_search_tool()
    ]
    agent = create_agent(
        model,
        system_prompt=prompts.RESEARCHER.SYSTEM_PROMPT,
        tools=researcher_tools,
        response_format=ProviderStrategy(schema=schemas.ResearchNotesSchema)
    ).with_retry(
        stop_after_attempt=3,
        exponential_jitter_params={
            "initial": 0.2,
            "max": 4,
            "exp_base": 2,
            "jitter": 0.1
        }
    )
    return agent


async def get_writer_agent():
    """
    Initializes and returns a writer agent with memory capabilities.
    """
    load_dotenv()
    writer_tools = [ 
        *await agent_mcp.get_mcp_tools(),
        agent_tools.get_web_search_tool(),
        agent_tools.count_words
    ]
    model = init_chat_model(
        "gpt-5-mini", 
        temperature=0.2
    )
    agent = create_agent(
        model, 
        system_prompt=prompts.WRITER.SYSTEM_PROMPT,
        tools=writer_tools,
    ).with_retry(
        stop_after_attempt=3,
        exponential_jitter_params={
            "initial": 0.2,
            "max": 4,
            "exp_base": 2,
            "jitter": 0.1
        }
    )
    return agent

async def get_critic_agent():
    """
    Initializes and returns a critic agent with memory capabilities.
    """
    load_dotenv()
    critic_tools = [ 
        *await agent_mcp.get_mcp_tools(),
        agent_tools.get_web_search_tool()
    ]
    model = init_chat_model(
        "gpt-5-mini",
        temperature=0.2
    )
    agent = create_agent(
        model,
        system_prompt=prompts.CRITIC.SYSTEM_PROMPT,
        tools=critic_tools,
        response_format=ProviderStrategy(schema=schemas.CriticSchema)
    ).with_retry(
        stop_after_attempt=3,
        exponential_jitter_params={
            "initial": 0.2,
            "max": 4,
            "exp_base": 2,
            "jitter": 0.1
        }
    )
    return agent