from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from typing import Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

# HuggingFace Model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0
)

model = ChatHuggingFace(llm=llm)

# Schema
class Review(BaseModel):

    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )

    summary: str = Field(
        description="A brief summary of the review"
    )

    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review either positive or negative"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list"
    )

    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer"
    )


# Output parser
parser = PydanticOutputParser(pydantic_object=Review)

# Prompt template
prompt = PromptTemplate(
    template="""
Extract the following information from the review.

{format_instructions}

Review:
{review}
""",
    input_variables=["review"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Chain
chain = prompt | model | parser


result = chain.invoke({
    "review": """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos.

The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches.

The 200MP camera is stunning with great zoom.

However, the phone is heavy and difficult to use with one hand. 
Samsung’s One UI also comes with bloatware and the $1300 price is very expensive.

Pros:
Insanely powerful processor
Stunning camera
Long battery life
S-Pen support

Review by Nitish Singh
"""
})

print(result)