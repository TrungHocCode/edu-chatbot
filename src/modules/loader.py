from langchain_community.document_loaders import TextLoader

def load_documents(filepath: str):
    loader=TextLoader(filepath, encoding="utf-8")
    return loader.load()