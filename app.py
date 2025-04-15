import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,    
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    AIMessagePromptTemplate,
)


load_dotenv()\
    
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("DevMind Chatbot")
st.subheader("Hi! I'm DevMind Chatbot. How can I assist you today?")

with st.sidebar:
    st.header("Chatbot Configuration")
    selected_model = st.selectbox("Choose model", ["deepseek-r1-distill-llama-70b"])
    st.markdown("### Chatbot capabilities:")
    capabilities = [
        "General Conversation",
        "Python Expert",
        "Debugging Assistant",
        "Code Documentation",
        "Solution Design",
        "Information Retrieval",
        
    ]
    st.multiselect("Select capabilities", capabilities, default=capabilities)
    with st.expander("### Quick Tips"):
        st.markdown("""
        - ** Tip 1 **: Use the chatbot for general conversation to explore its versatility.
        - ** Tip 2 **: Leverage the Python expertise for coding help and debugging.
        - ** Tip 3 **: Utilize the information retrieval for quick access to data and facts.
        """)
        
llm_engine = ChatGroq(
    api_key=groq_api_key,
    model=selected_model,
    temperature=0.3,
    )
    
def build_system_prompt(selected_capabilities):
    capabilities_text =", ".join(selected_capabilities)
    return f"You are a versatile AI chatbot with the following capabilities: {capabilities_text}. Engage in generating code, debugging, and providing information based on the user's request."
    
system_prompt = SystemMessagePromptTemplate.from_template(
    build_system_prompt(capabilities)
    )

if "message_log" not in st.session_state:
    st.session_state.message_log = []
        
chat_container = st.container()
    
with chat_container:
    for msg in st.session_state.message_log:
        with st.chat_message(msg["role"]):
            if "<think>" in msg["content"] and "</think>" in msg["content"]:
                start_idx = msg["content"].find("<think>") + len("<think>")
                end_idx=  msg["content"].find("</think>")
                think_content= msg["content"][start_idx:end_idx].strip()
                actual_response= msg["content"][end_idx + len("<think>"):].strip()
                with st.expander("AI Thought Process"):
                    st.markdown(think_content)
                st.markdown(actual_response)
            else:
                st.markdown(msg["content"])
                
user_query =  st.chat_input("Type your question or topic here...")

def generate_ai_response(prompt_chain, user_input):
    processing_pipeline= prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({"input":user_input})

def build_prompt_chain():
    prompt_sequence = [system_prompt]   
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)
    
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("Processing"):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain, user_query)
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()
    
            
                 
                    
                    

                    
                    
                
                
    

