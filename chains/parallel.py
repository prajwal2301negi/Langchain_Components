from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()

# HuggingFace models
llm1 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0
)

llm2 = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.2
)

model1 = ChatHuggingFace(llm=llm1)
model2 = ChatHuggingFace(llm=llm2)

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \nnotes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Parallel chain
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

# Merge chain
merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function.

The disadvantages include overfitting risk and expensive probability estimation.
"""

result = chain.invoke({"text": text})

print(result)

# Print graph
chain.get_graph().print_ascii()



# **Support Vector Machines (SVMs)**

# - **What are SVMs?**
#   - Supervised learning methods for classification, regression, and outlier detection.

# - **Advantages:**
#   - Effective in high-dimensional spaces.
#   - Still effective when dimensions > samples.
#   - Memory efficient (uses support vectors).
#   - Versatile (uses different kernel functions).

# - **Disadvantages:**
#   - Risk of overfitting.
#   - Expensive for probability estimation.

# **Quiz:**

# 1. **What are support vector machines used for?**
#    - Support vector machines (SVMs) are used for classification, regression, and outliers detection.

# 2. **Why are SVMs effective in high-dimensional spaces?**
#    - SVMs are effective in high-dimensional spaces because they can handle the complexity of data in such dimensions well. 

# 3. **Can SVMs still be effective when the number of dimensions is greater than the number of samples?**
#    - Yes, SVMs can still be effective in cases where the number of dimensions is greater than the number of samples.       

# 4. **What makes SVMs memory efficient?** 
#    - SVMs are memory efficient because they use a subset of training points, called support vectors, in the decision function.

# 5. **What are the potential drawbacks of using SVMs?**
#    - The potential drawbacks of using SVMs include the risk of overfitting and the expense of probability estimation.      