from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
from typing import Annotated
import requests
import json

# tool create
@tool
def get_conversion_factor(from_currency: str, to_currency: str) -> float:
    """Get conversion factor from one currency to another."""
    url = f"https://v6.exchangerate-api.com/v6/c34751a790387fe67df89e95/pair/{from_currency}/{to_currency}"
    response = requests.get(url)
    return response.json()
    # return data["rates"][to_currency]

result = get_conversion_factor.invoke({"from_currency": "USD", "to_currency": "INR"})
print("Conversion factor:", result)

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """Convert base currency value to target currency using conversion rate."""
    return base_currency_value * conversion_rate

# # tool binding
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage("What is the conversion fector from USD to INR and based on that can you convert 10 USD to INR?")]

ai_message = llm_with_tools.invoke(messages)
messages.append(ai_message)
# print(ai_message.tool_calls)

for tool_call in ai_message.tool_calls:
    #execute the first tool and get the value of conversion rate
    if tool_call['name'] == "get_conversion_factor":
        get_conversion_factor_result = get_conversion_factor.invoke(tool_call)
        
        # fetch this conversion rate
        conversion_rate = json.loads(get_conversion_factor_result.content)["conversion_rate"]
        print("Conversion rate:", conversion_rate)
        
        messages.append(get_conversion_factor_result)
    #execute the second tool and get the converted rate from tool 1
    
    if tool_call['name'] == "convert":
        # fetch the current args
        tool_call['args']['conversion_rate'] = conversion_rate
        convert_result = convert.invoke(tool_call)
        messages.append(convert_result)

# print("messages",messages)

final_result = llm_with_tools.invoke(messages)
print("Final result:", final_result.content)