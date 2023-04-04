import json
import os
from typing import Sequence

from aleph_alpha_client import (
    Client,
    CompletionRequest,
    Document,
    Prompt,
    SemanticEmbeddingRequest,
    SemanticRepresentation,
    SummarizationRequest,
)
from loguru import logger
from pypdf import PdfReader
from scipy.spatial.distance import cosine


def get_client(token):
    return Client(token=token)


def embedd_documents(file_path: str, token: str):
    print(file_path)
    # get the client
    client = get_client(token=token)

    # select all of the file links
    files = os.listdir(file_path)

    # create result dict
    result_dict = {}
    # loop over the files
    for tmp_file in files:
        print("tmp FILE", tmp_file)
        # read the pdf file
        reader = PdfReader(os.path.join(file_path, tmp_file))
        number_of_pages = len(reader.pages)
        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            # save the text to a file
            # with open(f"data/output/{tmp_file}_{str(i)}.txt", "w") as f:
            # f.write(text)
            request = SemanticEmbeddingRequest(prompt=Prompt.from_text(text), representation=SemanticRepresentation.Document)
            api_result = client.semantic_embed(request, model="luminous-base")

            # push the example to the dict
            result_dict[f"{tmp_file}___{str(i)}"] = api_result.embedding

    # save the result dict in the output folder
    with open("data/output/embedding_dict.json", "w") as f:
        json.dump(result_dict, f)


def search_documents(query: str, token: str, top: int = 3):
    client = get_client(token=token)
    logger.debug("Starting Search Documents")
    request = SemanticEmbeddingRequest(prompt=Prompt.from_text(query), representation=SemanticRepresentation.Query)
    result = client.semantic_embed(request, model="luminous-base")
    logger.debug("Query was embedded")
    # load the embedding dict
    with open("data/output/embedding_dict.json") as f:
        embedding_dict = json.load(f)

    scoring_dict = {}
    # iterate over the dict
    for key, value in embedding_dict.items():
        scoring_dict[key] = cosine_similarity(result.embedding, value)

    logger.debug("Scoring was done")

    # find the key with the lowest value
    sorted_dict = {k: v for k, v in sorted(scoring_dict.items(), key=lambda item: item[1], reverse=True)}
    logger.debug("SUCCESS: Sorting Finished")
    top_result = list(sorted_dict.keys())[0]
    logger.debug("SUCCESS: Top Result Found", top_result)
    logger.debug(top_result)
    # get the top result

    # split the top_result after the last _ and then combine the rest
    top_result_name = top_result.split("___")[:-1]

    # get all the numbers from the top result
    top_result_numbers = [int(s) for s in top_result.split("___")[-1] if s.isdigit()]
    # combine the numbers to one number
    top_result_numbers = int("".join(map(str, top_result_numbers)))

    logger.debug(f"SUCCESS: Top Result Numbers Found: {top_result_numbers}")
    top_result_name = "".join(top_result_name)
    logger.debug(f"SUCCESS: Top Result Name Found: {top_result_name}")

    # recombine the top result

    logger.debug(f"SUCCESS: Top Result Name Found {top_result_name}")

    # load the document for the key and summarize it
    reader = PdfReader(os.path.join("data/input/", top_result_name))
    summary = ""
    results = []
    text_string = ""
    logger.debug("Process file")
    page = reader.pages[int(top_result_numbers)]
    text = page.extract_text()
    text_string += text
    results.append(generate_summary(text, client))
    logger.debug(results)

    # combine the results into one single string
    summary = "\n".join(results)
    logger.debug("SUMMARY: ", summary)

    # now combine the dokument and the question to a prompt
    prompt = Prompt.from_text(
        f"""
    This is a Question answering service in german.
    {text_string}
    ###
    Frage: {query}
    Antwort:
    """
    )

    request = CompletionRequest(prompt=prompt, maximum_tokens=200, stop_sequences=["\n", "###"])
    response = client.complete(request, model="luminous-extended")

    return text, response.completions[0].completion


# function to calculate similarity
def cosine_similarity(first_vector: Sequence[float], second_vector: Sequence[float]) -> float:
    return 1 - cosine(first_vector, second_vector)


# function for getting a summary
def generate_summary(text: str, client: Client) -> str:
    """This method generates a summary of a text.

    In this method a text is summarized using the Luminous Large Language Model
    Endpoint of Aleph Alpha. The api is called using the clients Summarization Request
    :param text: The text which should be summarized.
    :type text: str
    :param client: The initialized Aleph Alpha API Client.
    :type client: Client
    :return: Result of the summarization.
    :rtype: str
    """
    request = SummarizationRequest(document=Document.from_text(text))
    result = client.summarize(request, model="luminous-extended")
    return result.summary


# function that splits text by paragraphs
def split_text(text: str):
    return text.split("\n\n")
