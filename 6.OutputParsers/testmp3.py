import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()


prompt = PromptTemplate.from_template(
    "Write a detailed 50 words report on in hindi: {topic}"
)


chain = prompt | model | parser


topic = "The impact of AI on education"
generated_text = chain.invoke({"topic": topic})
print("Generated Text:\n", generated_text)

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="shimmer",  # or alloy, echo, fable, nova, onyx
    input=generated_text,
)

output_file = "output.mp3"
with open(output_file, "wb") as f:
    f.write(response.content)

print(f"\n🎧 Audio saved as {output_file}")
