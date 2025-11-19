# Import necessary components from LangChain and LangGraph
# create_agent: creates an agent that can use the chat model
# init_chat_model: initializes a connection to a chat model (like GPT)
# InMemorySaver: stores conversation history in memory (RAM) so the agent remembers previous messages
# HumanMessage: represents a message from the user
# load_dotenv: loads environment variables from a .env file (including your API key)
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from dotenv import load_dotenv

# Load environment variables from the .env file
# This reads your OPENAI_API_KEY so you don't have to hard-code it in your script
load_dotenv()

# Initialize the chat model
# "gpt-5-nano" is the model name - this creates a connection to OpenAI's API
# The model is what actually generates the AI responses
model = init_chat_model("gpt-5-nano")

# Create an agent that wraps the model
# An agent is more powerful than just a model - it can maintain conversation history
# checkpointer=InMemorySaver() enables memory so the agent remembers past messages in a conversation
agent = create_agent(
    model
)

# Create the first user message
# HumanMessage() wraps the text in a format the agent understands
# We put it in a list because the agent expects a list of messages
user_message = [
    HumanMessage("Hello, how are you today?")
]

# Send the first message to the agent and get a response
# invoke() sends the message and waits for the AI to respond
# thread_id identifies this conversation - messages with the same thread_id share history
response = agent.invoke(
    {"messages": user_message},
    {"configurable": {"thread_id": "1"}}
)

# Print the complete first response
# This shows the full response object with all metadata
print("********************FIRST RESPONSE********************")
print(f"AI Response: {response}\n")

# Extract just the content of the last message (the AI's actual text response)
# response['messages'][-1] gets the last message in the conversation
# .content extracts just the text content without metadata
print (f"AI Response Content: {response['messages'][-1].content}\n\n\n")

# Create a second user message
# Notice we don't include the first message here - the agent remembers it automatically
# because we're using the same thread_id
user_second_message = HumanMessage("Can you tell me a joke?")

# Send the second message by appending the new message to the previous messages
second_response = agent.invoke(
    {"messages": [*response["messages"], user_second_message]},
)

# Print the complete second response
print("********************SECOND RESPONSE********************")
print(f"AI Second Response: {second_response}\n")

# Extract just the content of the AI's second response
print (f"AI Second Response Content: {second_response['messages'][-1].content}")