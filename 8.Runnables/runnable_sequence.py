from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

prompt1 = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="Explain the following joke - {text}",
    input_variables=["text"],
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = RunnableSequence(prompt1 | model | parser | prompt2 | model | parser)

result = chain.invoke({"topic": "AI"})
print("Result: ", result)
print("Graph: ", chain.get_graph().print_ascii())
