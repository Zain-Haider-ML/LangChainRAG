from resources.components.create_embeddings import *
from resources.components.create_vector_store import create_chroma_vector_store
from resources.constant import *
from resources.components.querying_vector_store import query_vector_store_similarity_score_threshold
from resources.user_input.user_input_splitting_strategy import main_menu




# MAIN
store_name = main_menu()
# print(store_name)
embeddings = create_HuggingFaceEmbeddings()
# BERTEmbeddings = create_sentence_transfomers_embeddings()
# create_vector_store(file_path, 'chroma_db', STEmbeddings)
# create_chroma_vector_store(data_path, db_dir, 'chroma_db_' + store_name, embeddings, splitting_strategy = store_name)
query = "How did Juliet die?"
query_vector_store_similarity_score_threshold(query, db_dir, 'chroma_db_' + store_name, embeddings, k = 3, score_threshold = 0.4)
