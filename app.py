import streamlit as st

from src.model.Model import Model

st.title("Kelompok 4 test code")
with st.form("Form"):
    Smoke_history = st.selectbox(label="Smoking", options=['never','current','former','ever','not current'])
    bmi = st.number_input(label="bmi")
    HbA1c_level = st.number_input(label='HbA1c_level')
    blood_glucose_level = st.number_input(label='blood_glucose_level')
    submit = st.form_submit_button("Submit")
    
if submit:
    model = Model(data=[Smoke_history,bmi,HbA1c_level,blood_glucose_level])
    hasil = model.predict()
    st.write(hasil)