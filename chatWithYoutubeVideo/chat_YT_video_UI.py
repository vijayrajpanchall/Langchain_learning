import streamlit as st
import os
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
import re

# Page configuration
st.set_page_config(
    page_title="YouTube Transcript RAG",
    page_icon="🎥",
    layout="wide"
)

# Title and description
st.title("🎥 YouTube Transcript RAG Assistant")
st.markdown("**Analyze YouTube videos using AI-powered question answering**")

# Sidebar for configuration
st.sidebar.header("Configuration")

# OpenAI API Key input
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key"
)

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Advanced settings
with st.sidebar.expander("Advanced Settings"):
    chunk_size = st.slider("Chunk Size", 300, 1000, 500)
    chunk_overlap = st.slider("Chunk Overlap", 50, 200, 100)
    retrieval_k = st.slider("Number of Retrieved Chunks", 3, 10, 6)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.1)

# Helper functions
def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return if already an ID"""
    if "youtube.com" in url_or_id or "youtu.be" in url_or_id:
        # Extract ID from URL
        if "youtu.be/" in url_or_id:
            return url_or_id.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in url_or_id:
            return url_or_id.split("watch?v=")[1].split("&")[0]
    return url_or_id

def get_youtube_transcript(video_id):
    """Get transcript from YouTube video"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        return transcript, None
    except TranscriptsDisabled:
        return None, "No captions available for this video"
    except Exception as e:
        return None, f"Error fetching transcript: {str(e)}"

def format_docs(retrieved_docs):
    """Format retrieved documents for context"""
    if not retrieved_docs:
        return "No relevant context found."
    
    context_texts = []
    for doc in retrieved_docs:
        if hasattr(doc, 'metadata') and 'score' in doc.metadata:
            context_texts.append((doc.page_content, doc.metadata['score']))
        else:
            context_texts.append((doc.page_content, 0))
    
    # Sort by score and join
    context_texts.sort(key=lambda x: x[1], reverse=True)
    return "\n\n".join([text for text, _ in context_texts])

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'current_video_id' not in st.session_state:
    st.session_state.current_video_id = None

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📹 Video Processing")
    
    # Video input
    video_input = st.text_input(
        "YouTube Video URL or ID",
        placeholder="https://www.youtube.com/watch?v=VIDEO_ID or just VIDEO_ID",
        help="Enter a YouTube video URL or video ID"
    )
    
    process_button = st.button("🔄 Process Video", type="primary")
    
    if process_button and video_input and openai_api_key:
        video_id = extract_video_id(video_input)
        
        with st.spinner("Processing video transcript..."):
            # Get transcript
            transcript, error = get_youtube_transcript(video_id)
            
            if error:
                st.error(error)
            else:
                st.success("✅ Transcript fetched successfully!")
                
                # Display transcript preview
                with st.expander("📄 Transcript Preview"):
                    st.text_area("Transcript", transcript[:1000] + "..." if len(transcript) > 1000 else transcript, height=200)
                
                # Process transcript
                try:
                    # Text splitting
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", " ", ""]
                    )
                    
                    chunks = splitter.create_documents([transcript])
                    st.info(f"📊 Created {len(chunks)} text chunks")
                    
                    # Create embeddings and vector store
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                    vector_store = FAISS.from_documents(chunks, embeddings)
                    
                    # Store in session state
                    st.session_state.vector_store = vector_store
                    st.session_state.video_processed = True
                    st.session_state.current_video_id = video_id
                    
                    st.success("🎯 Vector store created successfully!")
                    
                except Exception as e:
                    st.error(f"Error processing transcript: {str(e)}")
    
    elif process_button and not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar")
    elif process_button and not video_input:
        st.warning("⚠️ Please enter a YouTube video URL or ID")

with col2:
    st.header("💬 Ask Questions")
    
    if st.session_state.video_processed and st.session_state.vector_store:
        st.success(f"✅ Ready to answer questions about video: {st.session_state.current_video_id}")
        
        # Question input
        question = st.text_area(
            "Your Question",
            placeholder="What are the main topics discussed in this video?",
            height=100
        )
        
        # Predefined questions
        st.markdown("**Quick Questions:**")
        col_q1, col_q2 = st.columns(2)
        
        with col_q1:
            if st.button("📋 Main Topics"):
                question = "What are the main topics discussed in this video?"
            if st.button("🔑 Key Points"):
                question = "What are the key points and takeaways?"
        
        with col_q2:
            if st.button("📊 Summary"):
                question = "Can you provide a comprehensive summary?"
            if st.button("💡 Insights"):
                question = "What are the most important insights?"
        
        ask_button = st.button("🤔 Ask Question", type="primary")
        
        if ask_button and question:
            try:
                with st.spinner("Thinking..."):
                    # Setup retriever
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": retrieval_k}
                    )
                    
                    # Setup LLM
                    llm = ChatOpenAI(
                        model="gpt-3.5-turbo", 
                        temperature=temperature, 
                        max_tokens=500
                    )
                    
                    # Setup prompt
                    prompt = PromptTemplate(
                        template="""You are a helpful assistant. Answer only from the provided
                        transcript context. If the context is insufficient, just say you don't know.
                        
                        Context: {context}
                        
                        Question: {question}
                        
                        Answer: Please provide a detailed response based only on the above context.
                        """,
                        input_variables=["context", "question"],
                    )
                    
                    # Setup chain
                    parallel_chain = RunnableParallel({
                        "context": retriever | RunnableLambda(format_docs),
                        "question": RunnablePassthrough(),
                    })
                    
                    parser = StrOutputParser()
                    main_chain = parallel_chain | prompt | llm | parser
                    
                    # Get answer
                    result = main_chain.invoke(question)
                    
                    st.markdown("### 🤖 Answer:")
                    st.write(result)
                    
                    # Show retrieved context
                    with st.expander("📚 Retrieved Context"):
                        retrieved_docs = retriever.invoke(question)
                        for i, doc in enumerate(retrieved_docs):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(doc.page_content)
                            st.markdown("---")
                    
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
        
        elif ask_button and not question:
            st.warning("⚠️ Please enter a question")
    
    else:
        st.info("👆 Please process a video first to start asking questions")

# Footer
st.markdown("---")
st.markdown(
    """
    **How to use:**
    1. Enter your OpenAI API key in the sidebar
    2. Paste a YouTube video URL or ID
    3. Click 'Process Video' to analyze the transcript
    4. Ask questions about the video content
    
    **Note:** Only videos with available captions/transcripts can be processed.
    """
)