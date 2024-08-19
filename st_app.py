import streamlit as st
from rag import rag_chain
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)




chat_history = []  # Collect chat history here (a sequence of messages)

# steamlit app
st.title("Conservational RAG")

user_info = st.text_input("Enter some information before your query:")

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