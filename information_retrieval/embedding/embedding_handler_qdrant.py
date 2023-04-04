# import os
# import qdrant_client
# from qdrant_client.http.models import Batch
# from omegaconf import DictConfig
# from utils.configuration_wrapper import load_config
# from aleph_alpha_client import ImagePrompt, Client, SemanticEmbeddingRequest, SemanticRepresentation, Prompt, SummarizationRequest, CompletionRequest, EvaluationRequest


# # @load_config("conf/main_conf.yml")
# # def get_client(cfg: DictConfig):
# #     return qdrant_client.QdrantClient("localhost", port=6333)

# # client = get_client()

# def embedd_documents(file_path: str):

#     # select all of the file links
#     files = os.listdir(file_path)

#     # create result dict
#     result_dict = {}
#     # loop over the files
#     for tmp_file in files:
#         request = SemanticEmbeddingRequest(prompt=Prompt.from_file(tmp_file), representation=SemanticRepresentation.Document)
#         api_result = client.semantic_embed(request, model="luminous-base")

#         # push the example to the dict
#         result_dict[tmp_file] = api_result.embedding

#         # # add the result into the database
#         # qdrant_client.upsert(
#         # collection_name="Example", points=Batch(ids=[1], vectors=[result.embedding],))

#     # save the result dict in the output folder


# def search_documents(query: str, top: int=3):
#     request = SemanticEmbeddingRequest(prompt=Prompt.from_text(query), representation=SemanticRepresentation.Query)
#     result = client.semantic_embed(request, model="luminous-base")
#     return client.search(collection_name="Example", query=result.embedding, top=top)

# # function to calculate similarity
# def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
#     return 1 - cosine(v1, v2)

# # function for getting a summary
# def generate_summary(text: str):
#     request = SummarizationRequest(prompt=Prompt.from_text(text))
#     result = client.summarize(request, model="luminous-extended")
#     return result.summary

# # function that splits text by paragraphs
# def split_text(text: str):
#     return text.split("\n\n")

# # function that evaluate two texts
# def evaluate(text1: str, text2: str):
#     request = EvaluationRequest(prompt=Prompt.from_text(text1), completion_expected=text2)
#     result = client.evaluate(request, model="luminous-extended")
#     return result
