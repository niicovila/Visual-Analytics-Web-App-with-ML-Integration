import streamlit as st

st.set_page_config(
page_title="Welcome",
page_icon="👋",
layout="wide",
initial_sidebar_state="expanded")
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to top, #00008b, #4b0082);
        opacity: 0.7;
        color: white;
    }
    [data-testid="stSidebar"] .sidebar-content {
        
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.header("Navigate through the different tabs to learn about all the features of this app")
st.sidebar.write(" 📢 In the welcome tab we will find a brief introduction to the use case: Vehicle Pricing. ")

#The title
st.title("Vehicle Pricing 🚗 ")

#The subheader
st.subheader("Introduction to the use case:")

#The text
st.write("We are one of the most popular car buying and selling platforms in the world. We are going to launch a new product based on a price recommender for users' vehicles. In this application you will be able to explore the data of vehicles advertised in the past, test the prediction model, and understand the model's decisions with the explainability tab.")


