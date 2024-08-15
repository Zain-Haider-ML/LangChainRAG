import os
# from langchain.text_splitter import CharacterTextSplitter
from resources.components.splitting_strategy import split_strategy
from langchain_chroma import Chroma
from resources.components.data_loader import CustomTextLoader
# from sentence_transformers import SentenceTransformer



def create_chroma_vector_store(data_path, db_dir, store_name, embeddings, splitting_strategy, ):
    persistent_directory = os.path.join(db_dir, store_name)
    # print('\n', file_path)
    # Check if the Chroma vector store already exists
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        # Ensure the text file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"The file {file_path} does not exist. Please check the path."
            )

        # List all text files in the directory
        data_files = [f for f in os.listdir(data_path) if f.endswith(".txt")]


        documents = []
        for data_file in data_files:
            file_path = os.path.join(data_path, data_file)
            loader = CustomTextLoader(file_path)
            data_docs = loader.load()
            for doc in data_docs:
                # Add metadata to each document indicating its source
                doc.metadata = {"source": data_file}
                documents.append(doc)

        # Split the document into chunks
        docs = split_strategy(strategy_name = splitting_strategy, documents = documents)

        # Display information about the split documents
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")
        # print(f"Sample chunk:\n{docs[0].page_content}\n")

        # Create the vector store and persist it automatically
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(
            docs, embeddings, persist_directory=persistent_directory)
        print("\n--- Finished creating vector store ---")

    else:
        print("Vector store already exists. No need to initialize.")