from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

# Step 1a: Indexing (Document Ingestion)
video_id = "o8NiE3XMPrM"  # Replace with your YouTube video ID
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video")

# Step 1b: Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Smaller chunks
    chunk_overlap=100,  # Smaller overlap
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

chunks = splitter.create_documents([transcript])

# print(len(chunks))

# Steps 1c & 1d: Indexing (Embedding Generation and Vector Store Creation)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(
    chunks, embeddings
)  # Create a vector store from the chunks
# print(vector_store.index_to_docstore_id)

# print(vector_store.get_by_ids(["8f261f9c-7b99-47a5-b618-adaccb570fda"]))

# step 2: Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
# print(retriever)

# Step 3: Querying
result = retriever.invoke("What is deepmind")
# print(result)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=500)
prompt = PromptTemplate(
    template="""You are a helpful assistant. Answer only from the provided
    transcript context. if the context is insufficient, just say you don't 
    Context: {context} 
    Question: {question}
    
    Answer: Please provide a detailed response based only on the above context.
    """,
    input_variables=["context", "question"],
)

question = "what was the key points discussed in this video? and what google brings new to complare to similar products?"
retrieved_docs = retriever.invoke(question)


def format_docs(retrieved_docs):
    if not retrieved_docs:
        return "No relevant context found."

    # Sort by relevance if score is available
    context_texts = []
    for doc in retrieved_docs:
        if hasattr(doc, "metadata") and "score" in doc.metadata:
            context_texts.append((doc.page_content, doc.metadata["score"]))
        else:
            context_texts.append((doc.page_content, 0))

    # Sort by score and join
    context_texts.sort(key=lambda x: x[1], reverse=True)
    return "\n\n".join([text for text, _ in context_texts])


# print(context_text)
parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
)

# result = parallel_chain.invoke('What is Google beam?');
# print(result)

parser = StrOutputParser()

main_chain = parallel_chain | prompt | llm | parser

result = main_chain.invoke("What are the main topics discussed in the video?")
print(result)
