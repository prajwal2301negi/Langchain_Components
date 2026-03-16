from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

import os
os.environ["USER_AGENT"] = "Mozilla/5.0"

# HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
Answer the following question

Question: {question}

From the following text:
{text}
""",
    input_variables=["question", "text"]
)

parser = StrOutputParser()

url = "https://en.wikipedia.org/wiki/Cricket"

loader = WebBaseLoader(url)

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({
    "question": "What is the product that we are talking about?",
    "text": docs[0].page_content
})

print(result)