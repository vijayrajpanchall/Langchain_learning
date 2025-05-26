from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests


# tool create
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# tool binding
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

llm_with_tools = llm.bind_tools([multiply])

query = HumanMessage("What is 3 times 4?")
messages = [query]

result = llm_with_tools.invoke(messages)
messages.append(result)

tool_result = multiply.invoke(result.tool_calls[0])


messages.append(tool_result)

# print(messages)

final_result = llm_with_tools.invoke(messages)
print("Final result:", final_result.content)
