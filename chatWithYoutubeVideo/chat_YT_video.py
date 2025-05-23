from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Step 1a: Indexing (Document Ingestion) 
video_id = "o8NiE3XMPrM"  # Replace with your YouTube video ID
try:
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    # Flatten it to plain text
    transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)
except TranscriptsDisabled:
    print("No captions available for this video")
    
# Step 1b: Indexing (Text Splitting)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript]) 

# print(len(chunks))  

# Steps 1c & 1d: Indexing (Embedding Generation and Vector Store Creation)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 
vector_store = FAISS.from_documents(chunks, embeddings)  # Create a vector store from the chunks
# print(vector_store.index_to_docstore_id)  

# print(vector_store.get_by_ids(["8f261f9c-7b99-47a5-b618-adaccb570fda"]))

# step 2: Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
# print(retriever)

# Step 3: Querying
result = retriever.invoke("What is deepmind");
# print(result)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
prompt = PromptTemplate(
    template="""You are a helpful assistant. Answer only from the provided
    transcript context. if the context is insufficient, just say you don't 
    Context: {context}\n\nQuestion: {question}""",
    input_variables=["context", "question"]
)

question = "what was the key points discussed in this video? and what google brings new to complare to similar products?"
retrieved_docs = retriever.invoke(question)

context_text = "\n".join([doc.page_content for doc in retrieved_docs])
# print(context_text)

final_prompt = prompt.invoke({
    "context": context_text,
    "question": question
})

answer = llm.invoke(final_prompt)
print(answer.content)