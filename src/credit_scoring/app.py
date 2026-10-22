import requests
import streamlit as st

st.title("Premium Credit Card")
st.write("New credit card with instant approvement")

with st.form("Apply now"):
    age = st.number_input("Your age", min_value=18)
    income = st.number_input("Your income (in thousands)", min_value=0)
    education = st.checkbox("I have a higher education")
    work = st.checkbox("I have a stable work")
    car = st.checkbox("I have a car")
    submit = st.form_submit_button("Apply")

if submit:
    data = {"age": age, "income": income, "education": education, "work": work, "car": car}
    response = requests.post("http://127.0.0.1:3333/score", json=data)
    if response.json()["approved"]:
        st.success("Your application has been approved!")
    else:
        st.success("We're recommending an alternative for you - debit card with 3% cashback")
