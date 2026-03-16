from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# HuggingFace LLM 
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=512
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content="You are a helpful AI assistant")
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))

    result = model.invoke(chat_history)

    chat_history.append(AIMessage(content=result.content))

    print("AI:", result.content)

print(chat_history)



# You: hii
# AI: Hello! How can I assist you today?
# You: tell me about NSUT
# AI: NSUT stands for Netaji Subhas University of Technology. It is a prestigious technical university located in Sector 16, Noida, Uttar Pradesh, India. Here are some key points about NSUT:

# 1. **Establishment**: NSUT was established in 1994 and was previously known as Netaji Subhas Institute of Technology (NSIT).

# 2. **Recognition**: It is recognized by the All India Council for Technical Education (AICTE) and is affiliated with the University of Delhi.

# 3. **Campus**: The university has a sprawling campus with modern infrastructure, including state-of-the-art laboratories, research centers, and sports facilities.

# 4. **Facilities**: NSUT offers various facilities for its students, including a library, computer centers, and student hostels.

# 5. **Academic Programs**: The university offers a wide range of undergraduate and postgraduate programs in engineering, information technology, architecture, management, and other allied branches of technology.

# 6. **Research**: NSUT is actively involved in research and development activities. It has several research centers and collaborations with industry and other academic institutions.

# 7. **Ranking and Recognition**: NSUT is consistently ranked high in national engineering college rankings. It is known for producing graduates who are highly sought after by top industries and research institutions.

# 8. **Exposure**: The university organizes various academic and cultural events, including seminars, workshops, and conferences, which provide students with exposure to the latest trends and advancements in their respective fields.

# If you have specific questions or need more detailed information about any aspect of NSUT, feel free to ask!
# You: exit