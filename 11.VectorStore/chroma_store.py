from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


from langchain.schema import Document


doc1 = Document(
    page_content="Virat Kohli is one of the most successful batsmen in IPL history. He has been a key player for RCB since the tournament began.",
    metadata={"team": "Royal Challengers Bangalore"},
)

doc2 = Document(
    page_content="Rohit Sharma is known for his exceptional captaincy and batting. He has led Mumbai Indians to multiple IPL titles.",
    metadata={"team": "Mumbai Indians"},
)

doc3 = Document(
    page_content="MS Dhoni is celebrated for his calm leadership and finishing skills. He has been the face of Chennai Super Kings for years.",
    metadata={"team": "Chennai Super Kings"},
)

doc4 = Document(
    page_content="Jasprit Bumrah is a world-class fast bowler known for his deadly yorkers. He has been a crucial part of Mumbai Indians' bowling attack.",
    metadata={"team": "Mumbai Indians"},
)

doc5 = Document(
    page_content="Ravindra Jadeja is an all-rounder who contributes with both bat and ball. He is a vital asset for Chennai Super Kings.",
    metadata={"team": "Chennai Super Kings"},
)

docs = [doc1, doc2, doc3, doc4, doc5]

vector_store = Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory="chroma_db",
    collection_name="ipl_players",
)

vector_store.add_documents(docs)

data = vector_store.get(include=["embeddings", "documents", "metadatas"])
print("Data: ", data.items())

# data = vector_store.similarity_search_with_score(
#     query="Who is the captain of Mumbai Indians?",
#     k=2,
# )
# print("Data: ", data)

# updated_doc1 = Document(
#     page_content="Virat Kohli is an Indian international cricketer who plays ODI cricket for the national team and is a former captain in all formats. He is a right-handed batsman and an occasional right-arm medium pace bowler.",
#     metadata={"team": "Royal Challengers Bangalore"}
# )
