#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install langchain


# In[2]:


#!pip install streamlit


# In[3]:


#!pip install PyPDF2


# In[4]:


#!pip install langchain-openai


# In[5]:


import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import openai

# API 키 검사 함수
def check_api_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception:
        return False

def process_text(text, api_key):
#CharacterTextSplitter를 사용하여 텍스트를 청크로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    if not chunks:
        return None

    #임베딩 처리(벡터 변환), 임베딩은 OpenAI 모델을 사용합니다.
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=api_key)
    documents = FAISS.from_texts(chunks, embeddings)
    return documents

def main():  #streamlit을 이용한 웹사이트 생성
    st.set_page_config(page_title="PDF 요약기", page_icon="📄")
    st.title("📄이충환의 PDF 요약하기")
    st.divider()
    
    # 사이드바 설정: API 입력용
    with st.sidebar:
        st.title("설정")
        # API type: API 키가 노출되지 않도록 **** 형태로 입력
        user_api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
    
    # 키 입력 여부에 따른 상태 메시지 표시
        if user_api_key:
            if check_api_key(user_api_key):
                st.success("✅ 연결되었습니다!")
            else:
                st.error("❌ 유효하지 않은 키입니다. 다시 확인해 주세요.")
        else:
            st.warning("🔑 API Key를 입력해 주세요.")
            
        st.markdown("[API Key 발급받기](https://platform.openai.com/api-keys)")

    # pdf 파일 업로드 버튼
    pdf = st.file_uploader('PDF파일을 업로드해주세요', type='pdf')

    if pdf is not None:
        # 키 검증이 실패하면 진행하지 않음
        if not user_api_key or not check_api_key(user_api_key):
            st.info("먼저 유효한 OpenAI API Key를 입력해 주세요.")
            st.stop() # 유효한 API 키가 없다면 코드 실행 즉시 중단
        
        pdf_reader = PdfReader(pdf)
        text = ""   # 텍스트 변수에 PDF 내용을 저장
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted

        documents = process_text(text, user_api_key)
        # 텍스트 추출 실패 처리
        if documents is None:
            st.error("PDF에서 요약할 수 있는 텍스트를 찾지 못했습니다. 스캔된 이미지인지 확인해 보세요.")
            st.stop()
        
        query = "업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."  # LLM에 PDF파일 요약 요청

        if query:
            docs = documents.similarity_search(query)
            llm = ChatOpenAI(model="gpt-3.5-turbo", api_key=user_api_key, temperature=0.1)
            chain = load_qa_chain(llm, chain_type='stuff')
            
            with st.spinner('PDF 내용을 분석하여 요약 중입니다...'): # 요약하는 중에 나오는 로딩 애니메이션
                with get_openai_callback() as cost:
                    response = chain.run(input_documents=docs, question=query)
                    print(cost)

            st.subheader('--요약 결과--:')
            st.write(response)
            st.caption(f"발생 비용: ${cost.total_cost:.4f}")

if __name__ == '__main__':
    main()

# In[ ]:




