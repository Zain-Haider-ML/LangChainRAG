import os
from langchain_chroma import Chroma



def query_vector_store_similarity_score_threshold(query, db_dir, store_name, embeddings, k = 3, score_threshold = 0.4):
    # Load the existing vector store with the embedding function
    persistent_directory = os.path.join(db_dir, store_name)
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

    # Retrieve relevant documents based on the query
    retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": k, "score_threshold": score_threshold},)
    relevant_docs = retriever.invoke(query)
    # relevant_docs = retriever.get_relevant_documents(query)

    # Display the relevant results with metadata
    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")