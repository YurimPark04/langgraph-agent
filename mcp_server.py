from mcp.server.fastmcp.prompts import base
from mcp.server.fastmcp import FastMCP
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from datetime import datetime
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate


# 환경변수 로드
load_dotenv()


# Create an MCP server
mcp = FastMCP("Demo", json_response=True)

# 벡터 스토어 생성 
embeddings_function = OpenAIEmbeddings(
    model="text-embedding-3-large"

)

vector_store = Chroma(
    embedding_function=embeddings_function,
    collection_name='real_estate_tax',
    persist_directory='./real_estate_tax_collection' 
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})  


rag_prompt = hub.pull('rlm/rag-prompt')
llm = ChatOpenAI(model="gpt-4o-mini")  # 답변이 잘 안나오는데는 mini 모델을 써서 그럴수도 있다. (원래 gpt-4o 사용)
small_llm = ChatOpenAI(model="gpt-4o-mini")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
# 공제액 계산 도구

tax_deductible_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | small_llm
    | StrOutputParser()
)

deductible_question = f'주택에 대한 종합부동산세 과세표준의 공제액을 알려주세요'
# 사용자별 공제액 계산을 위한 프롬프트 템플릿
user_deduction_prompt = """아래 [Context]는 주택에 대한 종합부동산세의 공제액에 관한 내용입니다. 
사용자의 질문을 통해서 가지고 있는 주택수에 대한 공제액이 얼마인지 금액만 반환해주세요

[Context]
{tax_deductible_response}

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
    | small_llm
    | StrOutputParser()
)

# MCP 서버로 연결
@mcp.tool(
    name='tax_deductible_tool',
    description=    """
    
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



tax_base_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | small_llm
    | StrOutputParser()
)

# 과세표준 계산 방법을 검색하기 위한 표준 질문 정의
tax_base_question = '주택에 대한 종합부동산세 과세표준을 계산하는 방법은 무엇인가요? 수식으로 표현해서 수식만 반환해주세요'



@mcp.tool(
    name='tax_base_tool',
    description="""종합부동산세 과세표준을 계산하기 위한 공식을 검색하고 형식화합니다.
    
    이 도구는 RAG(Retrieval Augmented Generation) 방식을 사용하여:
    1. 지식 베이스에서 과세표준 계산 규칙을 검색
    2. 검색한 규칙을 수학 공식으로 형식화

    Args:
        question (str): 사용자의 질문 (미리 정의된 질문이 사용됨)
        
    Returns:
        str: 과세표준 계산 공식
    """,
)
def tax_base_tool() -> str:
    
    # 미리 정의된 질문으로 tax_base_chain을 실행하여 계산 공식 획득
    tax_base_response = tax_base_chain.invoke(tax_base_question)
    return tax_base_response






# DuckDuckGo 검색 도구 초기화
search = DuckDuckGoSearchRun()

# 현재 공정시장가액비율 검색을 위한 함수
def get_market_value_rate_search():
    """
    현재 연도의 공정시장가액비율을 웹에서 검색합니다.
    
    Returns:
        str: 현재 공정시장가액비율 정보를 포함한 검색 결과
    """
    return search.invoke(f"{datetime.now().year}년도 공정시장가액비율은?")

market_value_rate_search = get_market_value_rate_search()

# 공정시장가액비율 추출을 위한 프롬프트 템플릿 정의
# Context에서 관련 정보를 추출하고 사용자 상황에 맞는 비율을 반환하도록 설계
market_value_rate_prompt = PromptTemplate.from_template("""아래 [Context]는 공정시장가액비율에 관한 내용입니다. 
당신에게 주어진 공정시장가액비율에 관한 내용을 기반으로, 사용자의 상황에 대한 공정시장가액비율을 알려주세요.
별도의 설명 없이 공정시장가액비율만 반환해주세요.

[Context]
{context}

[Question]
질문: {question}
답변: 
""")

# 공정시장가액비율 계산을 위한 체인 구성
# 프롬프트 -> LLM -> 문자열 파서 순으로 처리
market_value_rate_chain = (
    market_value_rate_prompt
    | small_llm
    | StrOutputParser()
)

# 공정시장가액비율 계산을 위한 커스텀 도구
@mcp.tool(
    name = 'market_value_rate_tool',
    description="""사용자의 부동산 상황에 적용되는 공정시장가액비율을 결정합니다.
    
    이 도구는:
    1. 현재 공정시장가액비율 정보가 포함된 검색 결과를 사용
    2. 사용자의 특정 상황(보유 부동산 수, 부동산 가치)을 분석
    3. 적절한 공정시장가액비율을 백분율로 반환

    Args:
        question (str): 부동산 소유에 대한 사용자의 질문
        
    Returns:
        str: 공정시장가액비율 백분율 (예: '60%', '45%')
    """,
)
def market_value_rate_tool(question: str) -> str:
    
    # 검색된 정보와 사용자 질문을 기반으로 공정시장가액비율 계산
    market_value_rate = market_value_rate_chain.invoke({
        'context': market_value_rate_search, 
        'question': question
    })
    return market_value_rate





@mcp.tool(
    name = 'house_tax_tool',
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
    """,
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



####################################
# MCP 프롬프트
####################################
@mcp.prompt(
    name='houst_tax_system_prompt',
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
    # MCP 에는 시스템 프롬프트가 존재하지 않음 (user하고 assistant만 존재)
    return base.UserMessage(content=system_message_content)


# Run with streamable HTTP transport
if __name__ == "__main__":
    mcp.run(transport="stdio")  # 로컬 파일 경로에 접근 (프록시 서버)