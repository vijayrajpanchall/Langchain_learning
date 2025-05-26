from langchain_community.document_loaders import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

loader = CSVLoader("Files/employees.csv", encoding="utf-8")

docs = loader.load()

print(docs[0])
print(len(docs))
