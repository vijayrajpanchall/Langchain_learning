from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# chat template

chat_template = ChatPromptTemplate(
    [
        ("system", "You are a helpful customer support agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}"),
    ]
)


chat_history = []
# Load chat history from a file
with open("chat_history.txt") as file:
    chat_history.extend(file.readlines())

print(chat_history)

# creaete a prompt
prompt = chat_template.invoke(
    {
        "chat_history": chat_history,
        "query": "What is the status of my order?",
    }
)

print(prompt)
