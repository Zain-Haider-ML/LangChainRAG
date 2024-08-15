from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

def split_strategy(strategy_name, documents):

    if strategy_name == 'SentenceTransformersTokenTextSplitter':
        sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
        docs = sent_splitter.split_documents(documents)
    
    elif strategy_name == 'TokenTextSplitter':
        token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
        docs = token_splitter.split_documents(documents)
    
    elif strategy_name == 'RecursiveCharacterTextSplitter':
        rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = rec_char_splitter.split_documents(documents)
    
    elif strategy_name == 'Custom':
        
        class CustomTextSplitter(TextSplitter):
            def split_text(self, text):
                # Custom logic for splitting text
                return text.split("\n\n")  # Example: split by paragraphs
        
        custom_splitter = CustomTextSplitter()
        docs = custom_splitter.split_documents(documents)
    
    else:
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
    
    return docs