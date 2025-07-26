from langchain.text_splitter import CharacterTextSplitter

def splitter_docs(docs):
    splitter=CharacterTextSplitter(chunk_size=500,chunk_overlap=30)
    return splitter.split_documents(docs)