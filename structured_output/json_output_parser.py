from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0
)

model = ChatHuggingFace(llm=llm)

# JSON schema
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg"],
      "description": "Return sentiment of the review either negative or positive"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {"type": "string"},
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

# JSON parser
parser = JsonOutputParser()

# Prompt template
prompt = PromptTemplate(
    template="""
Extract the following information from the review and return it as JSON.

Schema:
{schema}

Review:
{review}
""",
    input_variables=["review"],
    partial_variables={"schema": json_schema}
)

# Chain
chain = prompt | model | parser

result = chain.invoke({
    "review": """
I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! 
The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos.

The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often.

What really blew me away is the 200MP camera—the night mode is stunning.

However, the weight and size make it uncomfortable for one-handed use. 
Also, Samsung’s One UI still comes with bloatware.

The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor
Stunning 200MP camera
Long battery life
S-Pen support

Review by Nitish Singh
"""
})

print(result)