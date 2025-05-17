from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

url = "https://www.geeksforgeeks.org/"


model = ChatOpenAI()

prompt = PromptTemplate(
    template="Answer the following question \n  {question} from the following text \n  {text}",
    input_variables=['question','text'],
)

parser = StrOutputParser()

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({"question": "What is GeeksforGeeks?",
                   "text": docs[0].page_content}))
