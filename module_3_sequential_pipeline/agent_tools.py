from langchain.tools import tool

@tool
def count_words(text: str) -> int:
    """
    Counts the number of words in the given text.

    Args:
        text (str): The input text to count words from.

    Returns:
        int: The number of words in the text.
    """
    words = text.split()
    return len(words)

def get_web_search_tool() -> dict:
    """
    Returns the schema for OpenAI's built-in web search tool.

    Returns:
        str: A placeholder string indicating a web search was performed.
    """
    return {"type": "web_search"}

