import streamlit as st
import pandas as pd
import numpy as np

st.write("""
# Our super app
Hello *world!*
""")
data = pd.read_csv("project_watson/data/train.csv")

st.sidebar.header("Write two sentences")

def sort_language(idiom):
    return data.premise[data.language == idiom].reset_index(drop=True)[np.random.randint(0,len(idiom))]

# warning: our 2 sentences must be fixed when selecting one

def user_input_premise():
    premise = st.sidebar.selectbox(
        "Pick one sentence",
        (sort_language("English"),
        sort_language("Chinese"),
        sort_language("Arabic"),
        sort_language("French"),
        sort_language("Russian"))
        )

    prem_df = pd.DataFrame({"Pick one sentence" : premise}, index=[0])
    return prem_df

def user_input_hypothesis():

    hypothesis = st.sidebar.selectbox(
        "Pick another sentence",
        (sort_language("English"),
        sort_language("Chinese"),
        sort_language("Arabic"),
        sort_language("French"),
        sort_language("Russian"))
        )
    hyp_df = pd.DataFrame({"Pick another sentence" : hypothesis}, index=[0])
    return hyp_df

prem_df = user_input_premise()
hyp_df = user_input_hypothesis()


st.subheader('User Input parameters')
st.write(prem_df)
st.write(hyp_df)

st.subheader('Class labels')
st.write(pd.DataFrame(data={
    "label" : [0,1,2],
    "evaluation" : ["Entailment","Neutral","Contradiction"]
    }).set_index("label"))
