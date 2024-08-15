from langchain_community.document_loaders import TextLoader
from langchain.schema import Document

class CustomTextLoader(TextLoader):
    def lazy_load(self):
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                yield Document(page_content=content, metadata={"source": self.file_path})
        except UnicodeDecodeError:
            raise RuntimeError(f"Failed to decode the file {self.file_path} with UTF-8 encoding.")