import streamlit as st
from src.model.model import Model
from src.Rag.SemanticSearch import get_retriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

#config
load_dotenv()
st.set_page_config(page_title='Diabetes AI Assistant', layout='wide')

# st.title("Kelompok 4 test code")

#-- Sidebar buat prediction
with st.sidebar:
    st.header("diabesetes risk prediksyen")
    with st.form("Form"):
        Smoke_history = st.selectbox(label="Smoking", options=['never','current','former','ever','not current'])
        bmi = st.number_input(label="bmi")
        HbA1c_level = st.number_input(label='HbA1c_level')
        blood_glucose_level = st.number_input(label='blood_glucose_level')
        submit = st.form_submit_button("predikc")
    
    if submit:
        model = Model(data=[Smoke_history,bmi,HbA1c_level,blood_glucose_level])
        hasil = model.predict()
        st.write(hasil)

#--main chat area
st.title("Diabetes Health Assistant")

SYSTEM_PROMPT = """You are a diabetes health assistant. Answer ONLY based on the provided context from medical documents. 
If the answer is not in the context, say "I don't have enough information from the documents to answer this."
Be concise, accurate, and cite sources. Never give medical advice outside the provided documents."""

#session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retriever" not in st.session_state:
    st.session_state.retriever = get_retriever(k=5)
if "llm" not in st.sesstion_state:
    st.session_state.llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free"),
        api_key=os.getenv("LLM_API_KEY"),
        temprature=0
    )
if "rag_chain" not in st.session_state:
    prompt = ChatPromptTemplate.from_message([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}") 
    ])
    st.session_state.rag_chain = (
        {"Context": st.session_state.retriever, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

#reset button
col1, col2 = st.colums([6, 1])
with col2:
    if st.button("reset chat", type="secondary"):
        st.session_state.message = []
        st.rerun()

#chat histrory
for msg in st.session_state.message:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("source"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** (page {src['page']}) - score: {src['score']:.3f}")
                    st.text(src["preview"])

#chat input
if user_input := st.chat_input("Ask About Diabetes...."):

    st.session_state.message.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("searching documents...."):
            docs = st.session_state.retriever.invoke(user_input)
            sources = [ 
                {
                    "source": d.metadata.get("source", "unkown"),
                    "page": d.metadata.get("page", "N/A"),
                    "score": 1.0,
                    "preview": " ".join(d.page_content.split())[:300]
                }
                for d in docs
            ]
            #generate answer
            response = st.session_state.rag_chain.invoke(user_input)
            st.markdown(response)

            if sources:
                with st.expander("sources"):
                    for src in sources:
                        st.caption(f"**{src['source']}** (page {src['page']})")
                        st.text(src["preview"])

    st.session_state.message.append({
        "role": "assistant",
        "content": response,
        "sources": sources
        })