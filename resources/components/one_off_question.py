import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema import (
    HumanMessage,
    SystemMessage,
)

# if not os.getenv("HUGGINGFACEHUB_API_TOKEN"):
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = getpass.getpass("Enter your token: ")

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'


def OFQ(query, relevant_docs):

    print('\n One Off Question\n')

    llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=768,
    do_sample=False,
    repetition_penalty=1.03,)

    chat_model = ChatHuggingFace(llm = llm)

    combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)
    messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),   ]

    res = chat_model.invoke(messages)
    print("\nAnswer : ", res.content, '\n')

def OFQ_with_continual_chat(db, retriever):
    
    llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=768,
    do_sample=False,
    repetition_penalty=1.03,)

    # Contextualize question prompt
    # This system prompt helps the AI understand that it should reformulate the question
    # based on the chat history to make it a standalone question
    contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is.")

    # Create a prompt template for contextualizing questions
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]   )

    # Create a history-aware retriever
    # This uses the LLM to help reformulate the question based on chat history
    history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt  )

    qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}" )

    # Create a prompt template for answering questions
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
    )

    # Create a chain to combine documents for question answering
    # `create_stuff_documents_chain` feeds all retrieved context into the LLM
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Create a retrieval chain that combines the history-aware retriever and the question answering chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # print("Start chatting with the AI! Type 'exit' to end the conversation.")

        # while True:
        #     query = input("You: ")
        #     if query.lower() == "exit":
        #         break
        #     # Process the user's query through the retrieval chain
        #     result = rag_chain.invoke({"input": query, "chat_history": chat_history})
        #     # Display the AI's response
        #     print(f"AI: {result['answer']}")
        #     # Update the chat history
        #     chat_history.append(HumanMessage(content=query))
        #     chat_history.append(SystemMessage(content=result["answer"]))
    return rag_chain