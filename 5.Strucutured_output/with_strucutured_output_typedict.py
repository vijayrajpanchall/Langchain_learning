from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional, TypedDict, Annotated, Literal
from pydantic import BaseModel, Field

load_dotenv()
model = ChatOpenAI()

class Review(BaseModel):
    key_themes: list[str] = Field(description="Key themes in the review")
    
    summary: str = Field(description= "Summary of the review")
    sentiment: Literal["pos", "neg"] = Field(description="Sentiment of the review (positive, negative, neutral)");
    pros: Optional[list[str]] = Field(default=None, description="Pros of the product")
    cons: Optional[list[str]] = Field(default=None, description="Cons of the product")
    name: Optional[str] = Field(default=None, description="Write the name of the reviewer")
    
structured_model = model.with_structured_output(Review)
# write mobile review
result = structured_model.invoke("""I recently purchased the OnePlus 13R as a gift for my mom ahead of Mother’s Day, and it’s been a delightful experience so far. Right out of the box, the packaging felt premium – classic OnePlus style with the bold red theme and neatly arranged accessories.

First Impressions:
The setup was quick, and the phone felt super snappy from the start. The touch responsiveness is excellent – something my mom instantly noticed and appreciated. The display is vibrant, bright even in outdoor lighting, and the in-display fingerprint sensor works smoothly and reliably.
Camera:
One of the standout features is the telephoto camera. For the price, the detail and clarity in zoomed shots are genuinely impressive. My mom enjoys taking close-up photos of plants and family, and she’s already started experimenting with portrait and macro modes.
Reviewed by vijay raj panchal
""")

# print(result)
# print(result['summary'])
print(result.name)