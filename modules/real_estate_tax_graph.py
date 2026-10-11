from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import START, END
from langchain_tavily import TavilySearch
from datetime import date
from collections import Counter
from langsmith import Client

client = Client()

load_dotenv()

class AgentState(TypedDict):
    query: str  # 사용자 질문
    answer: str  # 세율 계산

    tax_base_equation: str # 과세표준 계산 수식
    tax_deduction: str # 공제액
    market_ratio: str # 공정시장가액비율
    tax_base: str  # 과세표준 계산

graph_builder = StateGraph(AgentState)


# langchain 의 text splitter : 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 100,
    separators=['\n\n', '\n']
)

loader = UnstructuredMarkdownLoader(
    "../documents/output_docs/real_estate_tax.md", 
    mode="elements", 
    strategy="fast")

# data = loader.load()
document_list = loader.load_and_split(text_splitter=text_splitter) 

# %%

embeddings = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)




vector_store = Chroma.from_documents(
    documents=document_list,
    embedding=embeddings,
    collection_name='real_estate_tax_collection',
    persist_directory='../real_estate_tax_collection'   # 로컬에 설정한 경로로 데이터가 남음 (DB의 역할을 한다)
)

retriever = vector_store.as_retriever(search_kwargs={'k':3})  # 3개 문서 반환

data = vector_store.get(include=["documents"])
documents = data["documents"]
content_counts = Counter(documents)
duplicate_groups = {
    content: count
    for content, count in content_counts.items()
    if count > 1
}

print(f"전체 레코드 수: {len(documents)}")
print(f"고유 문서 수: {len(content_counts)}")
print(f"중복된 문서 그룹 수: {len(duplicate_groups)}")
print(
    "중복으로 추가된 레코드 수:",
    sum(count - 1 for count in duplicate_groups.values()),
)


# %%
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model='exaone3.5:7.8b', 
    num_predict=512,
)


# 경량화된 모델
small_llm = ChatOllama(
    model = 'bge-m3:latest'  
)

# 프롬프트
rag_prompt = client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True,
)

query = '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'


# 과세표준 계산 수식

# retriever를 동적으로 활용
tax_base_retrieval_chain = (
    {'context' : retriever, 'question': RunnablePassthrough()} 
    | rag_prompt | llm | StrOutputParser()
)



tax_base_equation_prompt = ChatPromptTemplate.from_messages([
    ('system', '사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요.부연설명없이 수식만 리턴해주세요'),
    ('human', '{tax_base_equation_information}')
])


# tax_base_retrieval_chain 를 구동해서 나온 답변을 input 으로 넣어준다
tax_base_equation_chain = (
    {'tax_base_equation_information' : RunnablePassthrough()} 
    | tax_base_equation_prompt
    | llm
    | StrOutputParser()
)

tax_base_chain = (
    {'tax_base_equation_information': tax_base_retrieval_chain}
    | tax_base_equation_chain
)
##############################################################

def get_tax_base_equation(state: AgentState) -> str:
    tax_base_equation_question = '주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요'
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    return {'tax_base_equation':tax_base_equation}  # state 를 return 



# %%
tax_deduction_chain = (
    {'context' : retriever, 'question': RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)


def get_tax_deduction(state:AgentState):
    tax_deduction_question = '주택에 대한 종합부동산세 계산시 공제금액을 알려주세요'
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    return {'tax_deduction':tax_deduction}  
    

# 프롬프트 작성
tax_market_ratio_prompt = ChatPromptTemplate.from_messages([
    ('system', f'아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext:\n{{context}}'),
    ('human', '{query}')
])

tavily_search_tool = TavilySearch(
    max_results=5,
    search_depth = "advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

def get_market_ratio(state:AgentState):
    query= f'오늘날짜:({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇%인가요?'
    context = tavily_search_tool.invoke(query)
    print(f'context === {context}')

    tax_market_ratio_chain = (
        tax_market_ratio_prompt
        | llm
        | StrOutputParser()
    )
    
    market_ratio = tax_market_ratio_chain.invoke({'context':context, 'query':query})
    
    return {'market_ratio' : market_ratio} # result는 List 형태


tax_base_caculation_prompt = ChatPromptTemplate.from_messages(
    [
        ('system', """주어진 내용을 기반으로 과세표준을 계산해주세요

                    과세표준 계산 공식: {tax_base_equation}
                    공제금액: {tax_deduction}
                    공정시장가액비율: {market_ratio}"""),
        ('human', "사용자 주택 공시가격 정보: {query}")
    ]
)


def calculate_tax_base(state: AgentState):

    
    tax_base_equation = state['tax_base_equation']
    tax_deduction = state['tax_deduction']
    market_ratio = state['market_ratio']
    query = state['query']

    # chaining
    tax_base_caculation_chain = (
        tax_base_caculation_prompt 
        | llm 
        | StrOutputParser()
    )

    # invoke
    tax_base = tax_base_caculation_chain.invoke({
        'tax_base_equation' : tax_base_equation,
        'tax_deduction' : tax_deduction,
        'market_ratio' : market_ratio,
        'query' : query
    })

    print(f'tax_base === {tax_base}')  # 로그 (셀 output 에 가독성있게 찍힌다)
    return {'tax_base': tax_base}


tax_rate_caculation_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요
    
    종합부동산세 세율:{context}'''),
    ('human', '''과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요
    
    과세표준 : {tax_base}
    주택 수 : {query}'''),
])

# 세율 계산
def calculate_tax_rate(state: AgentState):
    query = state['query']  # 사용자 질문
    tax_base = state['tax_base'] # 과세표준
    context = retriever.invoke(query)  # 문서 반환

    tax_rate_chain = (
        tax_rate_caculation_prompt
        | llm
        | StrOutputParser()
    )
    tax_rate = tax_rate_chain.invoke({
        'context':context, 
        'tax_base' : tax_base, 
        'query':query
        
    })
    
    print(f'tax_rate === {tax_rate}')
    return {'answer' : tax_rate}

graph_builder = StateGraph(AgentState)

graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)


# 병렬처리
graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')

# 3개의 에이전트가 calculate_tax_base로 모인다
graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')

graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)

graph = graph_builder.compile()


