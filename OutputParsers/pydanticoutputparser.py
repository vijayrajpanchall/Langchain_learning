from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm) 

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(gt=18, description="Age of the person")
    city: str = Field(description="City of the person")
    
parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Give me the name, age and city of a fictional {place} person \n  {format_instructions}",
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()},
)

prompt = template.invoke({"place": "Indian"})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print("Result: ", final_result)