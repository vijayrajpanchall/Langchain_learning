from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

loader = PyPDFLoader("Files/book.pdf")

docs = loader.load()

print(docs[0].metadata)
print(len(docs))

# model = ChatOpenAI()

# prompt = PromptTemplate(
#     template="Write a summary of the following poem \n  {poem}",
#     input_variables=['poem'],
# )

# parser = StrOutputParser()


# # print(type(docs))
# # print(docs[0].page_content)
# # print(docs[0].metadata)

# chain = prompt | model | parser

# print(chain.invoke({"poem": docs[0].page_content}))
