import streamlit as st
import pandas as pd
import numpy as np

def user_input_language():
    idiom = st.sidebar.selectbox(
        "",
        ([
            "Arabic",
            "Bulgarian",
            "Chinese",
            "English",
            "French",
            "German",
            "Greek",
            "Hindi",
            "Russian",
            "Spanish",
            "Swahili",
            "Thai",
            "Turkish",
            "Urdu",
            "Vietnamese"
            ])
        )
    return idiom

def two_sentences(idiom):
    data = pd.read_csv("project_watson/data/train.csv")
    premise = data.premise[data.language == idiom].reset_index(drop=True)[np.random.randint(0,len(idiom))]
    hypothesis = data[data['premise'] == premise].hypothesis.values[0]
    return premise, hypothesis


def main():
    #TITLE
    st.write("""
    # Project Watson
    Hello *world!*
    """)
    #SLIDEBAR
    st.sidebar.header("Pick a language")
    idiom = user_input_language()

    #INPUT PARAMETERS
    st.subheader('User Input parameters')
    df_two_sentences = pd.DataFrame(two_sentences(idiom), index=["premise", "hypothesis"]).T
    st.table(df_two_sentences.assign(hack="").set_index("hack"))

    #CLASS LABELS
    st.subheader('Class labels')
    st.write(pd.DataFrame(data={
        "label" : [0,1,2],
        "evaluation" : ["Entailment","Neutral","Contradiction"]
        }).set_index("evaluation"))

if __name__ == "__main__":
    main()
