from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("What is the capital of France?")
print(results)

llm = ChatOpenAI()

llm.invoke("hi")

# Pull the ReAct prompt from the hub
prompt = hub.pull("hwchase17/react")  # ReAct = Reasining + action

# step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm,
    tools=[search_tool],
    prompt=prompt,
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool],
    verbose=True,
    return_intermediate_steps=True,
)

# Step 5: Invoke the agent
response = agent_executor.invoke({"input": "3 ways to reach goa from jaipur"})
print(response)
print(response["output"])
