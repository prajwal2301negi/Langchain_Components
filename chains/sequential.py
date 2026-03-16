from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

# HuggingFace model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Placement in NSUT campus'})

print(result)




# ### 5-Pointer Summary of NSIT Placement Report (2022-2023)

# 1. **Placement Process Overview**:
#    - The process includes Pre-Placement Talks (PPTs), campus recruitment drives, mock interviews, and workshops.
#    - The placement cell provides comprehensive support to students.

# 2. **Key Statistics**:
#    - Total students: 4,500
#    - Total placements: 2,200
#    - Average CTC: INR 5.5 LPA
#    - Top employers: Google, Amazon, Microsoft, Adobe, Flipkart, and others
#    - Top industries: IT, ITES, BFSI, Automotive, and Aerospace

# 3. **Placement Statistics by Branch**:
#    - Computer Science and Engineering (CSE): 600 placements (70% of total)
#    - Electronics and Communication Engineering (ECE): 400 placements (18% of total)
#    - Mechanical Engineering (ME): 200 placements (9% of total)
#    - Electrical and Electronics Engineering (EEE): 150 placements (7% of total)
#    - Civil Engineering (CE): 50 placements (2% of total)

# 4. **Top Employers**:
#    - Google
#    - Amazon
#    - Microsoft
#    - Adobe
#    - Flipkart
#    - TCS
#    - Wipro
#    - Infosys
#    - Capgemini
#    - IBM

# 5. **Placement Cell Support**:
#    - Offers guidance, resources, and support throughout the placement process.  