from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("ipl news?")

# print("Search Results:", results)

shell_tool = ShellTool()
result = shell_tool.invoke("ls")
print("Shell Tool Result:", result)