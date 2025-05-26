from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(
        page_content="MMR helps you get diverse results when doing similarity search."
    ),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 1,
        # "similarity_threshold": 0.5,
    },
)

query = "What is LangChain?"
results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"Document {i+1}:")
    print(f"Content: {doc.page_content}")
    print("\n")
