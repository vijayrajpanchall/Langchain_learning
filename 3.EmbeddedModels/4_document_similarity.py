from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Tony stark is a billionaire. He is the owner of Stark Industries.",
    "Steve Rogers is a soldier. He is the owner of Stark Industries.",
    "Bruce Wayne is a billionaire. He is the owner of Wayne Enterprises.",
    "Clark Kent is a journalist. He is the owner of Daily Planet.",
    "Peter Parker is a photographer. He is the owner of Daily Bugle.",
]

query = "Who is the owner of Stark Industries?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]
 
print(query)
print(f"Document: {documents[index]}")
print(f"Score: {score}")