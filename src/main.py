from modules.loader import load_documents
from modules.splitter import splitter_docs
from modules.embedding import get_vectorstore
from modules.rag_chain import build_qa_chain
from langchain_ollama import OllamaLLM
import streamlit as st

def main():
    st.title("Chatbot RAG với LLaMA3")

    # Clear button
    if st.button("Xóa hội thoại"):
        st.session_state.messages = []

    # Khởi tạo messages nếu chưa có
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Hiển thị các tin nhắn cũ
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Tải dữ liệu & mô hình (nên cache để không load lại nhiều lần)
    @st.cache_resource(show_spinner="🔄 Đang tải mô hình và dữ liệu...")
    def load_rag_chain():
        file_path = "raw_data/data.txt"
        model = OllamaLLM(model="llama3")
        docs = load_documents(file_path)
        chunks = splitter_docs(docs)
        vectorstore = get_vectorstore(chunks)
        return build_qa_chain(vectorstore, model)

    qa_chain = load_rag_chain()

    # Nhận input người dùng
    if prompt := st.chat_input("Nhập câu hỏi..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Gọi mô hình RAG để trả lời
        response = qa_chain.invoke(prompt)

        st.chat_message("assistant").markdown(response['result'])
        st.session_state.messages.append({"role": "assistant", "content": response['result']})

if __name__ == "__main__":
    main()
