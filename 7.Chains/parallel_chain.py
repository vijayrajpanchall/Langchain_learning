from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
)

llm2 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-30B-A3B",
    task="text-generation",
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short questions answers from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n {notes} \n {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser,
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Qwen3 is the latest generation of large language models in Qwen series, offering a comprehensive suite of dense and mixture-of-experts (MoE) models. Built upon extensive training, Qwen3 delivers groundbreaking advancements in reasoning, instruction-following, agent capabilities, and multilingual support, with the following key features:

Uniquely support of seamless switching between thinking mode (for complex logical reasoning, math, and coding) and non-thinking mode (for efficient, general-purpose dialogue) within single model, ensuring optimal performance across various scenarios.
Significantly enhancement in its reasoning capabilities, surpassing previous QwQ (in thinking mode) and Qwen2.5 instruct models (in non-thinking mode) on mathematics, code generation, and commonsense logical reasoning.
Superior human preference alignment, excelling in creative writing, role-playing, multi-turn dialogues, and instruction following, to deliver a more natural, engaging, and immersive conversational experience.
Expertise in agent capabilities, enabling precise integration with external tools in both thinking and unthinking modes and achieving leading performance among open-source models in complex agent-based tasks.
Support of 100+ languages and dialects with strong capabilities for multilingual instruction following and translation.
"""

result = chain.invoke({"text": text})
print("Result: ", result)
print("Graph: ", chain.get_graph().print_ascii())
