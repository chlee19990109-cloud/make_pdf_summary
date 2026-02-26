# 📄 PDF 요약 웹사이트 (make_pdf_summary)

💡 **웹 서비스 링크:** [PDF 요약기 Streamlit App](https://makepdfsummary-enuwggqvu8l6svggovpwve.streamlit.app/)  
👨‍💻 **작성자:** 이충환

본 프로젝트는 사용자가 업로드한 PDF 파일의 내용을 분석하고, 핵심 내용을 3~5문장으로 빠르게 요약해 주는 AI 기반 웹 서비스입니다. **LangChain**과 **OpenAI LLM**을 활용하여 RAG(Retrieval-Augmented Generation) 파이프라인을 구축하였으며, **Streamlit**을 통해 사용자 친화적인 웹 인터페이스를 제공합니다. (참고: **랭체인으로 RAG 구현하기 - 5.3절 PDF 요약웹사이트 만들기**)

본 프로젝트는 교재(5.3절)의 가이드를 바탕으로 하되, **최신 LangChain 생태계(v0.2.x)** 및 **OpenAI의 최신 모듈(`langchain-openai`)**에 맞추어 모델 및 코드를 한 층 더 업그레이드하여 구현되었습니다.

---

## 🛠 상세 기술 요소 및 아키텍처 (Technical Details)

프로젝트 소스코드(`PDF_요약_웹사이트_만들기.py`)에 적용된 주요 기술과 상세 구현 로직은 다음과 같습니다.

### 1. 언어 모델 및 임베딩 (Large Language Model & Embeddings)
* **LLM (텍스트 요약 모델):** `ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)`
  * PDF 원본 교재에서는 `gpt-3.5-turbo-16k`를 예시로 들었으나, 본 프로젝트에서는 호환성 및 안정성을 위해 `gpt-3.5-turbo` 기반으로 구현했습니다.
  * 모델의 무작위성을 줄이고, 창의적인 텍스트 생성보다는 **문서 내용 기반의 정확한 사실 요약**을 도출하기 위해 `temperature=0.1` 옵션으로 낮게 설정했습니다.
* **텍스트 임베딩 모델 (Text Embeddings):** `OpenAIEmbeddings(model="text-embedding-ada-002")`
  * 원본 교재에서는 허깅페이스 오픈소스 모델(`all-MiniLM-L6-v2`)을 썼으나, 요약 정확성 향상 및 OpenAI API의 일관된 통합을 목적으로 OpenAI의 고성능 임베딩인 `text-embedding-ada-002`를 적용했습니다.
* **요약 체인 구조 및 프롬프트:** `load_qa_chain(llm, chain_type='stuff')`
  * **프롬프트 쿼리:** `"업로드된 PDF 파일의 내용을 약 3~5문장으로 요약해주세요."`
  * 질문과 연관도(Similarity)가 높은 것으로 검색된 텍스트 청크(Chunk)들을 프롬프트의 Context로 한 번에 모두 삽입(`stuff` 방식)하여 LLM에게 요약을 요청합니다.
* **비용 추적 (Cost Tracking):** `get_openai_callback()`
  * 랭체인의 콜백 메커니즘을 사용해 API 호출(질의 및 요약) 시 발생한 토큰 사용량 및 총 발생 비용을 실시간으로 계산합니다. 콜백 스코프 내 반환된 `cost.total_cost` 속성을 사용하여, 화면에 소수점 4자리 달러 단위로 투명하게 캡션 출력(`st.caption`)합니다.

### 2. 문서 파싱 및 벡터 검색 (Document Parsing & Vector Search)
* **PDF 파싱:** `PyPDF2.PdfReader`
  * 업로드된 PDF 파일의 페이지를 순회하며 `extract_text()` 메서드를 통해 컴퓨터가 읽을 수 있는 순수 텍스트 데이터를 차례대로 이어 붙여 누적 추출합니다.
* **텍스트 분할 (Text Splitter):** `RecursiveCharacterTextSplitter`
  * 원본 교재의 단순 `CharacterTextSplitter` 대신, 의미론적 문단 보존에 유리한 재귀적 개행 문자 분할 방식(`RecursiveCharacterTextSplitter`)을 도입했습니다.
  * `chunk_size = 1000`: 임베딩할 때 문맥을 충분히 담을 수 있도록 각 텍스트 덩어리를 1000자 기준으로 제한합니다.
  * `chunk_overlap = 200`: 텍스트를 나눌 때 앞뒤로 200자씩 겹치게(Overlap) 분할하여, 문장이나 문맥이 칼같이 단절되어 검색 누락이 발생하는 것을 원천 방지합니다.
* **벡터 데이터베이스 (Vector DB):** `FAISS` (Facebook AI Similarity Search)
  * 강력한 인메모리 방식의 벡터 검색 엔진입니다.
  * 잘게 쪼개진 텍스트 청크들을 벡터 매트릭스로 메모리상에 구성(`FAISS.from_texts()`)하고, 사용자의 요약 요청 쿼리를 벡터화한 뒤, 이 벡터맵 공간에서 코사인 유사도(Cosine Similarity) 알고리즘으로 가장 관련성이 높은 문서 부분들을 빠르게 색인해 옵니다.

### 3. 웹 프레임워크 및 UI 설계 (Streamlit Web Framework)
* **웹앱 라이브러리:** `Streamlit`
* **UI/UX 상세 구현 로직:**
  * **전역 설정:** `st.set_page_config`를 통해 브라우저 탭 타이틀("PDF 요약기")과 아이콘("📄")을 동기화하고 레이아웃을 정의합니다.
  * **사이드바 구성:** `st.sidebar` 컨텍스트를 활용하여 사용자가 독립적으로 OpenAI API Key를 입력(`st.text_input`)할 수 있는 란을 별도 공간에 두었습니다. 입력 타입은 `type="password"` 속성을 적용해 민감 정보가 렌더링되지 않도록 보안 처리했습니다.
  * **API 키 동적 유효성 검증:** 입력된 API Key가 실제로 수신 가능한 활성 키인지 체크하기 위해 `openai.OpenAI(api_key=...).models.list()` 요청 구조를 통한 자체적인 오류 핸들링(`check_api_key` 함수)을 거칩니다. 결과에 따라 실시간으로 `st.success`, `st.error`, `st.warning` 상태 바를 알림으로 띄워줍니다.
  * **인터랙션 피드백 (Spinner & Stop):** 파일 업로드 후, LLM 작업의 대기 시간이 길어질 때 사용자가 화면 멈춤으로 오해하지 않게끔 `st.spinner('PDF 내용을 분석하여 요약 중입니다...')`로 시각적 작업 중 표시 애니메이션을 제공합니다. 키 미제출 또는 텍스트 추출 불가 등 에러 사항 시엔 `st.stop()`으로 코드 실행 흐름을 즉시 차단, 사용자에게 방어적으로 가이드를 줍니다.

---

## ⚙️ 구체적 환경 설정 및 배포 (Environment Setup)

클라우드 호스팅(Streamlit Share 플랫폼) 및 로컬 디버깅 환경 모두에서 정확하고 완벽히 동작하도록 구축한 아키텍처 환경 설정입니다.

### 1. 배포 환경(Deploy Environment) 정보
- **배포 플랫폼:** Streamlit Community Cloud
- **실행 Python 버전:** `Python 3.11`
  * 파이썬 3.11 버전은 하부 CPython 실행 속도 최적화가 이루어진 메이저 버전이며, Streamlit 클라우드 환경에서 LangChain 모듈 간 충돌(의존성 지옥) 없이 가장 매끄럽고 안정적으로 구동됩니다.

### 2. 세부 패키지 요구사항 (`requirements.txt` 상세 정보)
최신 LangChain 생태계의 모듈 모음 간 상호 호환성을 100% 보장하기 위해 버전을 하드하게 고정(`==`)하여 구성했습니다.

```text
# 1. Frontend & Backend Web App Engine
streamlit                   # 웹 서비스 프론트엔드 및 백엔드 실행 코어 프레임워크

# 2. Document Loaders & Processors
PyPDF2                      # PDF 문서 바이너리를 파싱하여 텍스트를 순수 추출하는 라이브러리

# 3. LangChain Ecosystem
langchain==0.2.17           # 통합 RAG 및 LLM 체인 형성 파이프라인 프레임워크
langchain-openai==0.1.25    # 최신 OpenAI 모듈 통합 패키지 (ChatOpenAI, OpenAIEmbeddings 등 묶음)
langchain-community==0.2.19 # 써드파티 패키지 연동 지원 모듈 (FAISS 벡터 스토어 통합 기능 포함)

# 4. Vector Database & OpenAI
faiss-cpu                   # FAISS 데이터베이스의 코사인 유사도 벡터 연산 엔진 (가벼운 CPU 전용 버전)
openai                      # OpenAI 메인 API, API Key 유효성(client.models.list) 체크용 등으로도 쓰임
tiktoken                    # OpenAI 모델의 문맥 길이를 재기 위해 텍스트를 토큰화/인코딩 해주는 필수 플러그인
```

---

## 🚀 로컬 프로젝트 실행 가이드 (How to Run Locally)

1. **레포지토리 클론 및 작업 폴더 단말기 접근**
   ```bash
   git clone [자신의 GitHub 레퍼지토리 주소]
   cd make_pdf_summary-main
   ```

2. **파이썬 가상 환경(Virtual Environment) 구축 (필수 권장 사항)**
   설치 모듈 버전 충돌을 막기 위해 가상환경을 파고 접근합니다.
   ```bash
   # Python 3.11 버전을 기준으로 시스템 가상환경 생성
   python -m venv venv
   
   # 가상환경 활성화 (Windows)
   venv\Scripts\activate
   
   # 가상환경 활성화 (macOS / Linux)
   source venv/bin/activate
   ```

3. **라이브러리 패키지 설치**
   엄격한 버전 관리가 적용된 `requirements.txt`를 일괄 설치합니다.
   ```bash
   pip install -r requirements.txt
   ```

4. **웹 애플리케이션 데몬 실행**
   ```bash
   streamlit run PDF_요약_웹사이트_만들기.py
   ```

5. **실제 사용 가이드라인**
   - Streamlit 내부 서버가 구동되고 로컬 브라우저가 열리면(`http://localhost:8501`), **좌측 설정 사이드바**에 본인의 **OpenAI API Key**를 붙여넣기 합니다.
   - 키가 유효하다면 즉시 초록색으로 ✅ 연결 성공 알림이 나타납니다.
   - 우측 메인 화면의 대형 업로드 박스 안에 요약 분석하고자 하는 `.pdf` 문서를 드래그 앤 드롭 합니다.
   - 즉시 텍스트 청크화와 프로파일링 스피너가 구동되며, 몇 초 후 모델이 추려낸 3~5줄의 간결한 핵심 요약 결과문장이 브라우저에 찍힙니다. (소모된 토큰 비용 $\$$도 포함됩니다.)
