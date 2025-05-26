from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


text = """
You’ve likely interacted with large language models (LLMs), like the ones behind OpenAI’s ChatGPT, and experienced their remarkable ability to answer questions, summarize documents, write code, and much more. While LLMs are remarkable by themselves, with a little programming knowledge, you can leverage libraries like LangChain to create your own LLM-powered chatbots that can do just about anything.
I’m the CTO and co-founder of a startup. When we first started, we built a simple MVP website. Later, my CEO asked me to develop a complete web solution that included user, chef, and admin panels.
In an enterprise setting, one of the most popular ways to create an LLM-powered chatbot is through retrieval-augmented generation (RAG). When you design a RAG system, you use a retrieval model to retrieve relevant information, usually from a database or corpus, and provide this retrieved information to an LLM to generate contextually relevant responses.
Recently, my CEO has also started forcing me to attend his meetings some of which I have no interest in. This is taking away valuable time I need for coding.
"""

text_splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.5,
)

result = text_splitter.create_documents([text])
print("Result: ", len(result))
print("Result: ", result)
