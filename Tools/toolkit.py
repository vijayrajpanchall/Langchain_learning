from langchain_core.tools import tool

# Custom tools 
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

class mathToolkit:
    def getTools(self):
        return [add, subtract, multiply]
    
toolkit = mathToolkit()
tools = toolkit.getTools()

for tool in tools:
    print(tool.name + " => " + tool.description)