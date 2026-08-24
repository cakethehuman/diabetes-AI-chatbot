import streamlit as st

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.model.model import Model
from src.rag.SemanticSearch import get_retriever
from src.utils.Settings import settings

st.set_page_config(page_title='Diabetes AI Assistant', layout='wide')

if "proba" not in st.session_state:
    st.session_state.proba = None
if "retriever" not in st.session_state:
    st.session_state.retriever = get_retriever(k=5)
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        model=settings.LLM_MODEL,
        api_key=settings.LLM_API_KEY,
        base_url=settings.LLM_BASE_URL,
    )

# 2. Sidebar Prediction Form
with st.sidebar:
    st.header("Diabetes Risk Prediction")
    with st.form("Form"):
        Smoke_history = st.selectbox(
            label="Smoking", 
            options=['never', 'current', 'former', 'ever', 'not current']
        )
        bmi = st.number_input(label="BMI", min_value=0.0, format="%.2f")
        HbA1c_level = st.number_input(label="HbA1c Level", min_value=0.0, format="%.2f")
        blood_glucose_level = st.number_input(label="Blood Glucose Level", min_value=0.0, format="%.2f")
        submit = st.form_submit_button("Predict")

    if submit:
        model = Model(data=[Smoke_history, bmi, HbA1c_level, blood_glucose_level])
        proba, hasil = model._predict_proba()

        with st.container(border=True):
            st.text(f"Probability : {st.session_state.proba:.4f}")
            st.text(f"Predicted result : {st.session_state.hasil}")

st.title("Diabetes Health Assistant")

SYSTEM_PROMPT = """You are a diabetes health assistant. Answer ONLY based on the provided context from medical documents.
If the answer is not in the context, say "I don't have enough information from the documents to answer this."
Be concise, accurate, and cite sources. Never give medical advice outside the provided documents."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])
chain = prompt | st.session_state.llm | StrOutputParser()

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Reset Chat", type="secondary"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.caption(f"**{src['source']}** (page {src['page']})")
                    st.text(src["preview"])

if user_input := st.chat_input("Ask a question about diabetes or your risk factors..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # Single retrieval call
            docs = st.session_state.retriever.invoke(user_input)
            context_text = "\n\n".join(d.page_content for d in docs)
            
            sources = [
                {
                    "source": d.metadata.get("source", "unknown"),
                    "page": d.metadata.get("page", "N/A"),
                    "preview": " ".join(d.page_content.split())[:300]
                }
                for d in docs
            ]
            response = chain.invoke({
                "context": context_text,
                "question": user_input,
                "hasil": st.session_state.hasil if st.session_state.hasil is not None else "Not provided",
                "proba": f"{st.session_state.proba:.4f}" if st.session_state.proba is not None else "Not provided"
            })
            
            st.markdown(response)

            if sources:
                with st.expander("Sources"):
                    for src in sources:
                        st.caption(f"**{src['source']}** (page {src['page']})")
                        st.text(src["preview"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources
    })