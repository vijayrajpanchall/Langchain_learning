from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detailed report
template1 = PromptTemplate(
    template="Write a detailed report on the following topic: {topic}",
    input_variables=["topic"],
)

# 2nd prompt -> summary
template2 = PromptTemplate(
    template="Write a 5 line summary of the following text /n: {text}",
    input_variables=["text"],
)

prompt1 = template1.format(topic="Artificial Intelligence")

result1 = model.invoke(prompt1)
print(result1.content)

prompt2 = template2.invoke({'text': result1.content})
result2 = model.invoke(prompt2)
print(result2.content)