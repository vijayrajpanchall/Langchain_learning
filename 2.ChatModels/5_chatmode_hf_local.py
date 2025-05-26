from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os

load_dotenv()

# Debug prints
print(f"Token loaded: {'HUGGINGFACEHUB_API_TOKEN' in os.environ}")
print(f"Token value: {os.environ.get('HUGGINGFACEHUB_API_TOKEN', 'Not found')[:5]}...")

# Load model components directly
tokenizer = AutoTokenizer.from_pretrained(
    "gpt2", token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)
model = AutoModelForCausalLM.from_pretrained(
    "gpt2", token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

# Create the pipeline directly
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=10)

# Use the pipeline with LangChain
llm = HuggingFacePipeline(pipeline=pipe)
chat_model = ChatHuggingFace(llm=llm)

result = chat_model.invoke("What is the capital of India?")
print(result.content)
