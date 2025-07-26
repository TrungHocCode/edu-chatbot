from langchain.chains import RetrievalQA

def build_qa_chain(vectorstore,model):
    qa_chain = RetrievalQA.from_chain_type(llm=model, retriever=vectorstore.as_retriever())
    return qa_chain