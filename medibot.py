# ====== Imports ======
import os
import base64
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# ====== Environment Setup ======
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# ====== Background Image Setup ======
def get_base64_bg(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_bg(image_path):
    base64_img = get_base64_bg(image_path)
    ext = image_path.split(".")[-1]
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.9), rgba(0,0,0,0.9)),
                        url("data:image/{ext};base64,{base64_img}");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            color: white;
        }}
        .message {{
            border-radius: 10px;
            padding: 10px 15px;
            margin: 8px;
            max-width: 75%;
            font-size: 16px;
        }}
        .user-msg {{
            background-color: #d0e3ff;
            color: #003366;
            text-align: right;
            margin-left: auto;
        }}
        .bot-msg {{
            background-color: #ffffff;
            color: #222;
            text-align: left;
            margin-right: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ====== Set Background ======
set_bg("background.jpg")  # Your background image path

# ====== LangChain Setup ======
# Chat memory
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(
        k=2, memory_key="chat_history", return_messages=True
    )

# Vector DB and embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = Chroma(persist_directory="my_chroma_store", embedding_function=embeddings)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Prompt template
prompt_template = """
<s>[INST]
You are MediSachetak, a professional AI medical assistant. Your goal is to provide clear, medically accurate, and technically sound responses using appropriate medical terminology wherever necessary.

Instructions:
- Base your answers strictly on the provided CONTEXT and your medical knowledge.
- Use clinical or medical terms (e.g., "hypertension" instead of "high blood pressure") when relevant, but ensure explanations remain concise and understandable.
- Limit answers to **2–3 sentences** maximum.
- Always end with the disclaimer: 
  “Thankyou.”

If the question is outside your knowledge or CONTEXT, or not related to health, respond with:
“This assistant is designed to answer health-related questions only. Please ask a medically relevant question.”

CONTEXT: {context}

CHAT HISTORY: {chat_history}

USER QUESTION: {question}

YOUR ANSWER:
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# QA Chain
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=st.session_state.memory,
    retriever=db_retriever,
    combine_docs_chain_kwargs={'prompt': prompt}
)

# ====== Streamlit UI ======
st.markdown("<h1 style='text-align: center;'>🩺 MediGenie - Your Medical Genie🧙🏼‍♂️</h1>", unsafe_allow_html=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat container
chat_container = st.container()
input_container = st.container()

# ====== Show Chat Messages ======
with chat_container:
    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(
                f"<div class='message user-msg'>"
                f"<span style='font-size:20px;'>👤</span> <strong>You:</strong><br>{msg}"
                f"</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div class='message bot-msg'>"
                f"<span style='font-size:20px;'>👨‍⚕️</span> <strong>MediSachetak:</strong><br>{msg}"
                f"</div>",
                unsafe_allow_html=True
            )

# ====== Input Area ======
symptom_list = [
    "Fever", "Cough", "Headache", "Diarrhea", "Vomiting",
    "Chest Pain", "Shortness of Breath", "Fatigue", "Abdominal Pain"
]

with input_container:
    user_input = st.chat_input("Type your medical question here...")

    st.markdown("### Or select common symptoms:")

    selected_symptoms = st.multiselect(
        "Select symptoms:",
        symptom_list,
        key="symptoms",
        label_visibility="collapsed"
    )

    ask_button = st.button("🩺 Ask about selected symptoms", use_container_width=True)

    if ask_button:
        if selected_symptoms:
            combined_question = "What are the causes and treatments of " + ", ".join(selected_symptoms) + "?"
            with st.spinner("🤖 MediSachetak is thinking..."):
                result = qa.invoke(input=combined_question)
                st.session_state.chat_history.append(("You", combined_question))
                st.session_state.chat_history.append(("MediSachetak", result["answer"]))
                st.rerun()

# ====== Handle user input ======
if user_input:
    with st.spinner("🤖 MediSachetak is thinking..."):
        result = qa.invoke(input=user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("MediSachetak", result["answer"]))
        st.rerun()
