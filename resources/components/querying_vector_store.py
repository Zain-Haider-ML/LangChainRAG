import os
from langchain_chroma import Chroma



def query_vector_store(query, db_dir, store_name, embeddings, search_type, search_kwargs, ):
    # Load the existing vector store with the embedding function
    persistent_directory = os.path.join(db_dir, store_name)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,)
    relevant_docs = retriever.invoke(query)
    # relevant_docs = retriever.get_relevant_documents(query)

    # Display the relevant results with metadata
    # print("\n--- Relevant Documents ---")
    # for i, doc in enumerate(relevant_docs, 1):
    #     print(f"Document {i}:\n{doc.page_content}\n")
    #     if doc.metadata:
    #         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    return relevant_docs




def query_vector_store_for_continual_chat(db_dir, store_name, embeddings, search_type, search_kwargs, ):
    # Load the existing vector store with the embedding function
    persistent_directory = os.path.join(db_dir, store_name)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Retriever 
    retriever = db.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,)

    return db, retriever