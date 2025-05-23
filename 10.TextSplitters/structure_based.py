from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./Files/resume.pdf")

docs = loader.load()

text = """
You’ve likely interacted with large language models (LLMs), like the ones behind OpenAI’s ChatGPT, and experienced their remarkable ability to answer questions, summarize documents, write code, and much more. While LLMs are remarkable by themselves, with a little programming knowledge, you can leverage libraries like LangChain to create your own LLM-powered chatbots that can do just about anything.

In an enterprise setting, one of the most popular ways to create an LLM-powered chatbot is through retrieval-augmented generation (RAG). When you design a RAG system, you use a retrieval model to retrieve relevant information, usually from a database or corpus, and provide this retrieved information to an LLM to generate contextually relevant responses.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=10,
    chunk_overlap=0,  
)

result = splitter.split_documents(docs)
print("Result: ", len(result))
print("Result: ", result[0].page_content)