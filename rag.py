from langchain.schema import (
    HumanMessage,
    SystemMessage,
)
from resources.components.create_embeddings import *
from resources.components.create_vector_store import create_chroma_vector_store
from resources.constant import *
from resources.components.querying_vector_store import query_vector_store, query_vector_store_for_continual_chat
from resources.user_input.user_input_splitting_strategy import main_menu
from resources.components.one_off_question import OFQ, OFQ_with_continual_chat
import streamlit as st



# steamlit app
st.title("Conservational RAG")

store_name = st.text_input("Enter some information before your query:")

# store_name = main_menu()

if store_name:
    embeddings = create_HuggingFaceEmbeddings()
    # embeddings = create_SentenceTransformerEmbeddings()

    create_chroma_vector_store(data_path, db_dir, 'chroma_db_' + store_name, embeddings, splitting_strategy = store_name)
    # query = input('Enter query: ')

    search_kwargs = {"k": 3, "score_threshold": 0.4}
    # search_kwargs = {"k": 3, "fetch_k": 20, "lambda_mult": 0.5}
    search_type = 'similarity_score_threshold'
    # search_type = 'mmr'

    # relevant_docs = query_vector_store(query, db_dir, 'chroma_db_' + store_name, embeddings, search_type, search_kwargs)
    # OFQ(query, relevant_docs)

    db, retriever = query_vector_store_for_continual_chat(db_dir, 'chroma_db_' + store_name, embeddings, search_type, search_kwargs, )
    rag_chain = OFQ_with_continual_chat(db, retriever)




    chat_history = []  # Collect chat history here (a sequence of messages)


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Query..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        result = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
        print('\n', result['answer'])

        response = result['answer']
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        
        chat_history.append(HumanMessage(content=prompt))
        chat_history.append(SystemMessage(content=result["answer"]))

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please enter the required information before proceeding.")