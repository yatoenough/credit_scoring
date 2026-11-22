import os

import requests
import streamlit as st

st.title("Premium Credit Card")
st.write("New credit card with instant approvement")

with st.form("Apply now"):
    age = st.number_input("Your age", min_value=21, max_value=100)
    income = st.number_input("Your income (in thousands)", min_value=1)
    education = st.checkbox("I have a higher education")
    work = st.checkbox("I have a stable work")
    car = st.checkbox("I have a car")
    submit = st.form_submit_button("Apply")

if submit:
    data = {
        "age": age,
        "income": income,
        "education": education,
        "work": work,
        "car": car,
    }

    api_url = os.environ.get("API_URL", "http://127.0.0.1:3333")

    try:
        response = requests.post(api_url + "/score", json=data)
        response.raise_for_status()
        if response.json()["approved"]:
            st.success("Your application has been approved!")
        else:
            st.info(
                "We're recommending an alternative for you - debit card with 3% cashback"
            )
    except requests.Timeout:
        st.error("Request timed out after 5s.")
        st.stop()

    except requests.HTTPError as exc:
        if exc.response is not None:
            st.error(f"HTTP {exc.response.status_code}: {exc.response.reason}")
        else:
            st.error(f"HTTP error: {exc}")

    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        st.stop()
