from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path="Files", glob="*.pdf", loader_cls=PyPDFLoader)

# docs = loader.load()
docs = loader.lazy_load()

# print(docs[0].metadata)
for doc in docs:  # for lazy load learning
    print(doc.metadata)
# print(len(docs))
