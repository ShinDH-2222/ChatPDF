__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.callbacks.base import BaseCallbackHandler
import chromadb
import streamlit as st
import tempfile
import os
# from dotenv import load_dotenv
# load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")

# 파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!")
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# 업로드된 파일 처리
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,) # True로 설정하면 정규표현식으로 구분자를 인식합니다.
    texts = text_splitter.split_documents(pages)

    # Embedding
    embeddings_model = OpenAIEmbeddings(
        model = "text-embedding-3-large",
        # With the `text-embedding-3` class,
        # you can specify the embedding dimensions,
        # (e.g., `text-embedding-3-small` for 1024 dimensions.
    )

    # Vector Store Chroma DB
    chromadb.api.client.SharedSystemClient.clear_system_cache() # Chroma DB의 시스템 캐시를 지웁니다. (선택 사항)
    DB = Chroma.from_documents(texts, embeddings_model)

    # 스트리밍 처리할 Handler 생성
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text = initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text += token
            self.container.markdown(self.text)

    # User Input
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input("질문을 입력하세요: ")

    if st.button("질문하기"):
        with st.spinner("답변을 생성하는 중입니다..."):
            # Retriever (검색기)
            llm = ChatOpenAI(temperature=0) # 모델 지정 안하면 기본적으로 gpt-3.5-turbo 모델이 사용됩니다.
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=DB.as_retriever(), # Chroma DB에서 검색기를 가져옵니다.
                llm=llm # 검색 결과를 LLM으로 재처리하여 더 나은 결과를 얻습니다.
            ) # retriever_from_llm이 반환하는 것은 검색된 문서들의 객체 리스트입니다.

            # Prompt Template
            prompt = hub.pull("rlm/rag-prompt") # RAG (Retrieval-Augmented Generation) 프롬프트 템플릿을 허브에서 가져옵니다. 

            # Generate 
            chat_box = st.empty() # 답변이 출력될 컨테이너를 생성합니다.
            stream_handler = StreamHandler(chat_box) # 스트리밍 핸들러를 생성하여 답변이 출력될 컨테이너를 전달합니다.
            generate_llm = ChatOpenAI(temperature=0,
                                      streaming= True,
                                      callbacks=[stream_handler]) # 답변을 생성할 LLM을 생성할 때 스트리밍 핸들러를 콜백으로 전달합니다.

            def format_docs(docs):
                return "\n\n".join([doc.page_content for doc in docs]) # 객체 문서를 하나의 문자열로 합치는 함수입니다.
            rag_chain = (
                {"context" : retriever_from_llm | format_docs, # retriever_from_llm이 반환하는 객체 문서 리스트를 format_docs 함수를 통해 하나의 문자열로 변환하여 context로 사용합니다.
                "question" : RunnablePassthrough()} # question은 그대로 전달하는 RunnablePassthrough로 설정합니다.
                | prompt | generate_llm | StrOutputParser()
            )

            # 질문에 대한 답변 생성
            result = rag_chain.invoke(question)