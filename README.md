# **LangChainRAG - A Comprehensive Guide to RAG Implementation**

This repository is a comprehensive implementation of a Retrieval-Augmented Generation (RAG) system using LangChain. The project aims to cover the basics to advanced concepts of RAG by utilizing LangChain's capabilities.

## **Project Structure**

- **data**
  - Contains the text data files (`odyssey.txt`, `romeo_and_juliet.txt`) used for creating embeddings and vector stores. These files serve as the input data for the RAG pipeline.

- **resources**
  - **components**
    - `create_embeddings.py`: Script to generate embeddings from the input text data.
    - `create_vector_store.py`: Script to create a vector store using the generated embeddings.
    - `data_loader.py`: Handles the loading and preprocessing of text data.
    - `one_off_question.py`: Script to query the vector store with a single question.
    - `querying_vector_store.py`: Script to interact with the vector store and retrieve relevant text based on user queries.
    - `splitting_strategy.py`: Defines the strategies for splitting text data, such as RecursiveCharacterTextSplitter or SentenceTransformersTokenTextSplitter.
  - **constant**
    - Contains constant values and configurations used across the project.

- **user_input**
  - **user_input_splitting_strategy.py**: Allows the user to select and define the text-splitting strategy to be used for creating the vector store.

- **LICENSE**: Contains the licensing information for the project.

- **README.md**: The documentation file you're currently reading.

- **rag.py**: Main script to run the Streamlit app, which provides a user-friendly interface to interact with the RAG system.

- **requirements.txt**: Lists all the Python dependencies required to run the project.

- **st_app.py**: A supplementary script for setting up the Streamlit app environment.

## **Database Folder**

The `Database (db)` folder is not included in the repository. It will be automatically created when you run the code to build the vector store using the selected text-splitting strategy. Depending on the strategy chosen (e.g., `RecursiveCharacterTextSplitter`, `SentenceTransformersTokenTextSplitter`), the `db` folder will contain the corresponding vector store and metadata.

## **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/LangChainRAG.git
    cd LangChainRAG
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run rag.py
    ```

## **Usage**

1. Place your text files in the `data/` folder.
2. Define your text-splitting strategy in `user_input/user_input_splitting_strategy.py`.
3. Run `rag.py` to create embeddings, build the vector store, and launch the Streamlit app.
4. Use the Streamlit app to interact with your RAG system, query the vector store, and retrieve relevant information.

## **Contributing**

Feel free to contribute to this project by submitting issues or pull requests. Contributions are welcome!

## **License**

This project is licensed under the MIT License. See the `LICENSE` file for more details.

---