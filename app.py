import streamlit as st

from src.model.Model import Model

st.title("Diabetest prediction")
with st.form("Form for prediction"):
    Smoke_history = st.selectbox(label="Smoking", options=['never','current','former','ever','not current'])
    bmi = st.number_input(label="bmi")
    HbA1c_level = st.number_input(label='HbA1c_level')
    blood_glucose_level = st.number_input(label='blood_glucose_level')
    submit = st.form_submit_button("Submit")
    
if submit:
    mlModel = Model(data=[Smoke_history,bmi,HbA1c_level,blood_glucose_level])
    
    with st.container(border=True, horizontal_alignment='center'):
        proba, hasil = mlModel._predict_proba()
        st.text("Prediction result", text_alignment='center')
        st.text(hasil, text_alignment='center')
        st.text(f"confidence : {proba[0]:.3f}%")
    
    # if (hasil == 0):
    #     st.write("Not at all")
    # else:
    #     st.write("Diabetes")
        