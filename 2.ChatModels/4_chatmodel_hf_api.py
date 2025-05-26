from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")
print(result.content)

# llm = HuggingFaceEndpoint(
#     repo_id="mistralai/Mistral-Nemo-Base-2407",
#     provider="novita",
#     max_new_tokens=100,
#     do_sample=False,
# )
# print(llm.invoke("What is Deep Learning?"))
