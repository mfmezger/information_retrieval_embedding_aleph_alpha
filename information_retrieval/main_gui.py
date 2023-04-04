from pathlib import Path

import streamlit as st
from embedding.embedding_handler import embedd_documents, search_documents
from loguru import logger

# def setup_logger():
#     logger.add(sys.stdout, colorize=True, format="<green>{time}</green> <level>{message}</level>")
# setup_logger()

# set small icon in the tab bar
st.set_page_config(page_title="Information Retrieval Embedding Demo", page_icon=":mag:")


def start_embedding(file_path, token):
    embedd_documents(file_path, token)


# @load_config("conf/main_conf.yml") cfg: DictConfig
def initialize():
    save_path_input = "data/input/"

    # create the folder structure
    Path(save_path_input).mkdir(parents=True, exist_ok=True)

    Path("data/output").mkdir(parents=True, exist_ok=True)

    # create title
    st.title("Information Retrieval Embedding Demo")

    # The user needs to enter the aleph alpha api key
    aleph_alpha_api_key = st.text_input("Aleph Alpha Token", type="password")
    logger.debug("API Key was entered")
    # create a uploader for multiple files of the type pdf
    uploaded_files = st.file_uploader("Upload PDF Files", type="pdf", accept_multiple_files=True)

    # iterate over the Files
    for file in uploaded_files:
        logger.debug("File was uploaded")
        # save the files to the file system in the input folder
        with open(f"{save_path_input}{file.name}", "wb") as f:
            f.write(file.getbuffer())

    # create a button to start the embedding
    if st.button("Start Embedding"):
        logger.debug("Embedding was started")
        start_embedding(save_path_input, aleph_alpha_api_key)

    # create a textfield for the search query
    search_query = st.text_input("Search Query")
    # if the button search is clicked search
    if st.button("Start Search"):
        # search the documents
        logger.debug("Search was started")
        documents, qa = search_documents(query=search_query, token=aleph_alpha_api_key)
        # show the top 3 documents
        st.text_area("QA", value=qa)

        st.text_area("Document", value=documents, height=500)


# start the gui app
initialize()


# display at the bottom "if you encounter any problems please contact us at: marc.mezger@adesso.de"
st.markdown("If you encounter any problems please contact us at: marc.mezger@adesso.de")
