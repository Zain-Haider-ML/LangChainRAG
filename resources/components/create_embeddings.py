# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings


def create_HuggingFaceEmbeddings():
    # Create embeddings
    print("\n--- Creating embeddings ---")
    # embeddings = OpenAIEmbeddings(
    #     model="text-embedding-3-small")  
    # Update to a valid embedding model if needed
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("\n--- Finished creating embeddings ---")
    return embeddings


def create_SentenceTransformerEmbeddings():
    # Create embeddings
    print("\n--- Creating embeddings ---")  
    # Update to a valid embedding model if needed
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("\n--- Finished creating embeddings ---")
    return embeddings