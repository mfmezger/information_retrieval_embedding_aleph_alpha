from pathlib import Path

import streamlit as st


def initialize():

    # create the folder structure
    Path("../data/input").mkdir(parents=True, exist_ok=True)
    Path("../data/output").mkdir(parents=True, exist_ok=True)

    # create title
    st.title("Information Retrieval Embedding Demo")


# start the gui app
initialize()
