from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("./Files/resume.pdf")

# docs = loader.load()

text = """
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def __str__(self):
    return f"{self.name}({self.age})"
    
Objects can also contain methods. Methods in objects are functions that belong to the object.

Let us create a method in the Person class:
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=200,
    chunk_overlap=0,
)

result = splitter.split_text(text)
print("len: ", len(result))
print("Result: ", result[0])
