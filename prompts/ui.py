import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

# Load model
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

st.title("Simple LangChain Chatbot")

user_input = st.text_input("Enter your question")

if st.button("Ask"):
    
    messages = [
        SystemMessage(content="You are a helpful assistant"),
        HumanMessage(content=user_input)
    ]

    result = model.invoke(messages)

    st.write("### AI Response:")
    st.write(result.content)