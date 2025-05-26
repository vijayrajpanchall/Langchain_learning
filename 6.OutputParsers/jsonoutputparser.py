from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)
parser = JsonOutputParser()
# print("Parser format: ",parser.get_graph())

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n  {format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# prompt = template.format()
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser
result = chain.invoke({})


print("Result: ", result)
