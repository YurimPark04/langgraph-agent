from mcp.server import MCPServer
from pathlib import Path
import sys, os



# mcp/mcp_server.py 기준 프로젝트 루트
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DOCUMENT_PATH = (
    PROJECT_ROOT
    / "documents"
    / "output_docs"
    / "real_estate_tax.md"
)

PERSIST_DIRECTORY = PROJECT_ROOT / "real_estate_tax_collection"


from dotenv import load_dotenv
load_dotenv()


mcp = MCPServer("Demo")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@mcp.resource("greeting://{name}")
def greeting(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


# ipynb 파일에서 개발한 도구들을 서버로 이동
# 필요한 라이브러리들을 임포트
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langsmith import Client


# 문서 포맷팅을 위한 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

client = Client()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 100,
    separators=['\n\n', '\n']
)

loader = UnstructuredMarkdownLoader(
    str(DOCUMENT_PATH),  ## 수정
    mode="elements", 
    strategy="fast")

document_list = loader.load_and_split(text_splitter=text_splitter) 


embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)


# 리트리버 생성
vector_store = Chroma(
    collection_name="real_estate_tax_collection",
    persist_directory=str(PERSIST_DIRECTORY),
    embedding_function=embeddings,
)

retriever = vector_store.as_retriever(search_kwargs={'k':3})  # 3개 문서 반환


# 비용 최적화를 잘 하려면, 토큰을 적게 쓰거나 (==프롬프트 자체를 짧게 하거나)
# 모델을 작은 걸 써야 한다.
llm = ChatOllama(
    model='exaone3.5:7.8b', 
    num_predict=512,
)


# 경량화된 모델
small_llm = ChatOllama(
    model = 'llama3.1:latest'  
)

# 프롬프트
rag_prompt = client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True,
)


# 세금 공제액 정보를 검색하기 위한 체인 구성
tax_deductible_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | small_llm
    | StrOutputParser()
)

# 공제액 관련 기본 질문 정의
deductible_question = f'주택에 대한 종합부동산세 과세표준의 공제액을 알려주세요'

# 사용자별 공제액 계산을 위한 프롬프트 템플릿
user_deduction_prompt = """아래 [Context]는 주택에 대한 종합부동산세의 공제액에 관한 내용입니다. 
사용자의 질문을 통해서 가지고 있는 주택수에 대한 공제액이 얼마인지 금액만 반환해주세요

[Context]
{tax_deductible_response}  # 리트리버해서 가지고 온 결과 (배경지식; 사용자의 질문에 정보들을 기반으로 확실한 답변을  생성한다.)

[Question]
질문: {question}
답변: 
"""

# 프롬프트 템플릿 객체 생성
user_deduction_prompt_template = PromptTemplate(
    template=user_deduction_prompt,
    input_variables=['tax_deductible_response', 'question']
)

# 사용자별 공제액 계산을 위한 체인 구성
user_deduction_chain = (user_deduction_prompt_template
    | small_llm   # 작은 모델을 활용하여 비용을 최적화
    | StrOutputParser()
)

@mcp.tool(
        name='tax_deductible_tool',
        description="""사용자의 부동산 소유 현황에 대한 질문을 기반으로 세금 공제액을 계산합니다.
            
            이 도구는 다음 두 단계로 작동합니다:
            1. tax_deductible_chain을 사용하여 일반적인 세금 공제 규칙을 검색
            2. user_deduction_chain을 사용하여 사용자의 특정 상황에 규칙을 적용
        
            Args:
                question (str): 부동산 소유에 대한 사용자의 질문
                
            Returns:
                str: 세금 공제액 (예: '9억원', '12억원')
            """,
)
def tax_deductible_tool(question: str) -> str:
    
    # 일반적인 세금 공제 규칙 검색
    tax_deductible_response = tax_deductible_chain.invoke(deductible_question)
    
    # 사용자의 특정 상황에 규칙 적용
    tax_deductible = user_deduction_chain.invoke({
        'tax_deductible_response': tax_deductible_response, 
        'question': question
    })
    return tax_deductible



# 프롬프트 추가 
@mcp.prompt(
        name="house_tax_system_prompt",
        description="""종합부동산세 계산 프롬프트"""
)
def house_tax_system_prompt():

    system_message_content = """당신의 역할은 주택에 대한 종합부동산세를 계산하는 것입니다. 
        사용자의 질문이 들어오면, 사용자의 질문을 바탕으로 종합부동산세를 계산해주세요.
        종합부동산세를 계산하기 위해서는 과세표준을 어떻게 계산할지 파악해야하고, 
        사용자에 질문에 따른 공제액을 파악해야 하고, 
        사용자에 질문에 따른 공정시장가액비율을 파악해야 합니다.
        이 세가지를 파악하고 나면, 종합부동산세를 계산해주세요.
        """
    
    return system_message_content

@mcp.tool(
        name="house_tax_tool",
        description="""수집된 모든 정보를 사용하여 최종 종합부동산세액을 계산합니다.
            
            이 도구는 다음 정보들을 결합하여 최종 세액을 계산합니다:
            1. 과세표준 계산 공식
            2. 공정시장가액비율
            3. 공제액
            4. 세율표
        
            Args:
                tax_base_question (str): 과세표준 계산 공식
                market_value_rate_question (str): 공정시장가액비율
                tax_deductible_question (str): 공제액
                question (str): 부동산 세금 계산에 대한 사용자의 질문
                
            Returns:
                str: 설명이 포함된 최종 세금 계산액
            """
)
def house_tax_tool(tax_base_question: str, market_value_rate_question: str, tax_deductible_question: str, question: str) -> str:

    house_tax_prompt = ChatPromptTemplate.from_messages([
    ('system', f'''과세표준 계산방법: {tax_base_question}
    공정시장가액비율: {market_value_rate_question}
    공제액: {tax_deductible_question}

    위의 공식과 아래 세율에 관한 정보를 활용해서 세금을 계산해주세요.
    세율: {{tax_rate}}
    '''),
        ('human', '{question}')
    ])

    house_tax_chain = (
        {
            'tax_rate': retriever | format_docs,  # 벡터 DB에서 세율 정보 검색
            'question': RunnablePassthrough()     # 사용자 질문 그대로 전달
        }
        | house_tax_prompt    # 프롬프트 템플릿에 정보 전달
        | llm                 # LLM으로 계산 수행
        | StrOutputParser()   # 결과를 문자열로 변환
    )

    # 체인 실행하여 최종 세금 계산 결과 반환
    house_tax = house_tax_chain.invoke(question)
    return house_tax



if __name__ == "__main__":
    print("[startup] MCP 실행 시작", file=sys.stderr, flush=True)
    mcp.run(transport='stdio')  
