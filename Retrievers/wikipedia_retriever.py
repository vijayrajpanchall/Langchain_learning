from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=2, lang='en')

query = "The geopolitical history of India and Pakistan"
# result = retriever.get_relevant_documents(query)

docs = retriever.invoke(query)

for i, doc in enumerate(docs):
    print(f"Document {i+1}:")
    print(f"Title: {doc.metadata['title']}")
    print(f"Content: {doc.page_content}")
    # print(f"URL: {doc.metadata['url']}")
    print("\n")

# print("Result: ", docs)