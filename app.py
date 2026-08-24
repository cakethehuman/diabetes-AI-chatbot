import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.model.model import Model

from src.rag.SemanticSearch import get_retriever
from src.utils.Settings import settings

st.set_page_config(page_title='Diabetes AI Assistant', layout='wide')

#-- Sidebar buat prediction
with st.sidebar:
    st.header("Diabetes Risk Prediction")
    with st.form("Form"):
        Smoke_history = st.selectbox(label="Smoking", options=['never','current','former','ever','not current'])
        bmi = st.number_input(label="bmi")
        HbA1c_level = st.number_input(label='HbA1c_level')
        blood_glucose_level = st.number_input(label='blood_glucose_level')
        submit = st.form_submit_button("predict")
    
    if submit:
        model = Model(data=[Smoke_history,bmi,HbA1c_level,blood_glucose_level])
        proba, hasil = model._predict_proba()
        with st.container(border=True):
            st.text(f"Probability : {proba[0]:.3f}")
            st.text(f"Predicted result : {hasil}")

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
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )
if "rag_chain" not in st.session_state:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}") 
    ])
    st.session_state.rag_chain = (
        {"context": st.session_state.retriever, "question": RunnablePassthrough()}
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

#reset button
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("reset chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

#chat histrory
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("source"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** (page {src['page']}) - score: {src['score']:.3f}")
                    st.text(src["preview"])

#chat input
if user_input := st.chat_input("Ask About Diabetes...."):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("searching documents...."):
            docs = st.session_state.retriever.invoke(user_input)
            sources = [ 
                {
                    "source": d.metadata.get("source", "unknown"),
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

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
        })