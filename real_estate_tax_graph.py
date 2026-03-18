# %%
# from dotenv import load_dotenv
# load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph


class AgentState(TypedDict):
    query: str
    answer: str

    # 필요한 노드 ? 
    tax_base_equation: str # 과세표준 계산 수식 
    tax_deduction: str  # 공제액 
    market_ratio: str  # 공정시장가액비율
    tax_base: str   # 과세표준 계산
    

graph_builder = StateGraph(AgentState)

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, separators=["\n\n", "\n"])

# %%
from langchain_community.document_loaders import TextLoader

text_path = './docs/real_estate_tax.txt'
loader = TextLoader(text_path,  encoding='utf-8')  # ✅ 인코딩 명시! : 에러 발생함
document_list = loader.load_and_split(text_splitter)

# %%
# 과세표준을 계산하는 방법은 vector 스토어에 있다. retriever 부터 생성하자

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 벡터 스토어 생성 
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


vector_store = Chroma.from_documents(
    documents = document_list,
    embedding=embeddings,
    collection_name='real_estate_tax',
    persist_directory='./real_estate_tax_collection' 
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})  

# %%
query = '5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때 세금을 얼마나 내나요?'

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")  # 답변이 잘 안나오는데는 mini 모델을 써서 그럴수도 있다. (원래 gpt-4o 사용)
small_llm = ChatOpenAI(model="gpt-4o-mini")

# %%
from langchain_classic import hub

rag_prompt = hub.pull('rlm/rag-prompt')



# %%
# 수식을 가져오는 노드 생성
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

tax_base_equation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연설명 없이 수식만 리턴해주세요."),
        ("human", "{tax_base_equation_information}")
    ]
)

tax_base_retrieval_chain = (
    {'context': retriever, 'question' : RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)


tax_base_equation_chain = (
    {'tax_base_equation_information' : RunnablePassthrough()}
    | tax_base_equation_prompt
    | llm
    | StrOutputParser()
)

tax_base_chain = {'tax_base_equation_information' : tax_base_retrieval_chain} | tax_base_equation_chain

def get_tax_base_equation(state: AgentState):
    tax_base_equation_question = '주택에 대한 종합부동산세 계산 시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요'
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)    # chain 앞에 {'context': retriever} 가 추가되었으므로, context키는 삭제해도됨
    return {'tax_base_equation': tax_base_equation}


# %%
get_tax_base_equation({})

# 생성된 답변은 아래와 같다
# {'tax_base_equation': '주택에 대한 종합부동산세의 과세표준은 납세의무자가 소유한 주택의 공시가격 합산액에서 특정 금액을 공제한 후, 공정시장가액비율을 곱하여 계산합니다. 이 비율은 부동산 시장 동향과 지방 여건을 고려하여 60%에서 100% 사이로 대통령령으로 결정됩니다. 또한, 1세대 1주택자는 12억 원, 법인 또는 법인으로 보는 단체는 6억 원, 일반적으로는 9억 원까지의 합산액이 기준이 됩니다.'}

# 그런데, 줄글로 되어있으면 LLM이 알아듣기 힘들 수 도 있다. 
# 1. 그래서 query 에 "계산하는 방법을 수식으로 표현해서 알려주세요'" 라고 명확히 물어봤지만, 그래도 답변이 수식이 아니라 줄글로 나옴
# 2. 아래 셀을 생각해보자




# {'tax_base_equation': '과세표준 = (주택 공시가격 합산 - 공제금액) × 공정시장가액비율'}

# %%
# 2.
# LLM에 여러 태스크를 한번에 주면, LLM이 헤맨다
# 번역, 요약, 분석에 능하다. 그런데, "분석해서 요약해서 번역해줘 " -> 결과가 안나오는 이유


# chain을 하나 더 만들어서 tax_base_equation 으로 가져온 정보에 기반으로 수식만 추출하게끔 짠다

# %%
# 공제액을 찾는 노드


tax_deduction_chain = (
    {'context': retriever, 'question' : RunnablePassthrough()} 
    | rag_prompt 
    | llm 
    | StrOutputParser()
)

def get_tax_deduction(state: AgentState):
    tax_deduction_question = '주택에 대한 종합부동산세 계산 시 공제금액을 알려주세요'
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)    # chain 앞에 {'context': retriever} 가 추가되었으므로, context키는 삭제해도됨
    return {'tax_deduction': tax_deduction}


# %%
get_tax_deduction({})


# {'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제금액은 1세대 1주택자의 경우 12억원, 법인 및 법인으로 보는 단체는 6억원, 일반적인 경우 9억원입니다. 또한 만 60세 이상인 1세대 1주택자는 연령에 따라 20%, 30%, 40%의 공제율이 적용됩니다.'}

# %%
# 주택에 대한 공정시장가액비율
# 대통령령이라서 웹 서치를 해야함 


# web-search 노드
# from langchain_tavily import TavilySearch
from langchain_community.tools import TavilySearchResults   # cannot import 에러
# from langchain_community.tools import TavilySearch   # cannot import 에러
from datetime import date


tavily_search_tool = TavilySearchResults(
    max_results=5, 
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    search_depth="advanced",
    
)

tax_market_ratio_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"아래 정보를 기반으로 공정시장 가액비율을 계산해주세요\n\nContext:\n{{context}}"),
        ("human", "{query}")
    ]
)

def get_market_ratio(state: AgentState):
    query = f'오늘 날짜: ({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇 %인가요'
    context = tavily_search_tool.invoke(query)  
    print(f'context == {context}')
    tax_market_ratio_chain = tax_market_ratio_prompt | llm | StrOutputParser()
    market_ratio = tax_market_ratio_chain.invoke({'context': context, 'query': query})
    return {'market_ratio' : market_ratio}  


# %%
get_market_ratio({})

# {'market_ratio': '2026년 3월 18일 기준 주택의 공정시장가액비율은 60%로 설정되어 있습니다. 이는 법적으로 정해진 범위에 따른 것으로, 특정한 조건에 따라 다른 비율이 적용될 수도 있지만, 기본적인 설정은 60%입니다.'}
# 23년이 gpt 학습날짜 컷오프다. - 지금 24년도니까, 프롬프트에 "날짜정보"를 주도록 하자

# %%
from langchain_core.prompts import ChatPromptTemplate

tax_base_caculation_prompt = ChatPromptTemplate.from_messages([

    ('system', '''
        주어진 내용을 기반으로 과세표준을 계산해주세요
                                                                
        과세표준 계산 공식: {tax_base_equation}
        공제금액: {tax_deduction}
        공정시장가액비율: {market_ratio}
        사용자 주택 공시가격 정보 : {query}'''),

    ('human', '사용자 주택 공시가격 정보 : {query}')

])

    
    
    
    
    

def calculate_tax_base(state: AgentState):
    tax_base_equation = state['tax_base_equation']
    tax_deduction = state['tax_deduction']
    market_ratio = state['market_ratio']

    query = state['query']

    # 체인 생성
    tax_base_caculation_chain =    tax_base_caculation_prompt | llm | StrOutputParser()
    tax_base = tax_base_caculation_chain.invoke({
        'tax_base_equation': tax_base_equation,
        'tax_deduction': tax_deduction,
        'market_ratio': market_ratio,
        'query': query
    })
    print(f'tax_base == {tax_base}')
    return {'tax_base': tax_base}


# %%


# %%
initial_state = {
    'query': query,
    'tax_base_equation': '과세표준 = (주택 공시가격 합산 - 공제금액) × 공정시장가액비율',
    'tax_deduction': '주택에 대한 종합부동산세 계산 시 공제금액은 1세대 1주택자의 경우 12억원, 법인 및 법인으로 보는 단체는 6억원, 일반적인 경우 9억원입니다. 또한 만 60세 이상인 1세대 1주택자는 연령에 따라 20%, 30%, 40%의 공제율이 적용됩니다.',
    'market_ratio': '2026년 3월 18일 기준 주택의 공정시장가액비율은 60%로 설정되어 있습니다. 이는 법적으로 정해진 범위에 따른 것으로, 특정한 조건에 따라 다른 비율이 적용될 수도 있지만, 기본적인 설정은 60%입니다.'
    
    }

# %%
calculate_tax_base(initial_state)

# %%

tax_rate_caculation_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요
     종합부동산세 세율: {context}'''),

    ('human', '''과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요
     
    과세표준 : {tax_base}
    주택 수 : {query}''')

])

def calculate_tax_rate(state: AgentState):
    
    # retrieve -> 사용자 질문 기반 (세율 정보를 검색해서 가지고 옴) => [세율 정보]
    # 과세표준, [세율 정보]와 같이 LLM에 던진다 
    query = state['query']
    tax_base = state['tax_base']
    context = retriever.invoke(query)  # 사용자 질문
    
    tax_rate_chain = (
        tax_rate_caculation_prompt | llm | StrOutputParser()
    )

    tax_rate = tax_rate_chain.invoke({
        'context' : context,
        'tax_base':tax_base,
        'query' : query
    })

    print(f'tax_rate == {tax_rate}')
    return {'answer': tax_rate}




# %%
# {'tax_base': '주택 공시가격 정보를 바탕으로 과세표준을 계산해보겠습니다.\n\n1. **주택 공시가격 합산**:\n   - 5억 + 10억 + 20억 = 35억\n\n2. **공제금액 결정**:\n   - 일반적인 경우에 해당하므로 9억원을 공제합니다.\n\n3. **공제된 과세표준 계산**:\n   - 과세표준 = (주택 공시가격 합산 - 공제금액) × 공정시장가액비율\n   - 과세표준 = (35억 - 9억) × 60%\n   - 과세표준 = (26억) × 60% = 15.6억\n\n따라서, 이 사용자는 15.6억 원의 과세표준으로 종합부동산세를 계산하게 됩니다. \n\n이제 세금 산출을 위해 종합부동산세 세율을 적용해야 하는데, 실제 세율은 해당 과세표준에 따라 차등적으로 적용됩니다. 2022년 기준, 1세대 1주택자의 경우 1.0%의 세율이 일반적으로 적용되지만 세율은 변동할 수 있습니다.\n\n4. **종합부동산세 예시 계산**:\n   - 세금 = 과세표준 × 세율\n   - 세금 = 15.6억 × 1% = 1560만원\n\n결론적으로, 주택 공시가격이 35억원인 경우의 대략적인 종합부동산세는 약 1560만원입니다. \n(실제 세율은 변동할 수 있으므로, 정확한 계산은 최신 세법에 따라 확인해야 합니다.)'}
tax_base_state = {'tax_base': '주택 공시가격 정보를 바탕으로 과세표준을 계산해보겠습니다.\n\n1. **주택 공시가격 합산**:\n   - 5억 + 10억 + 20억 = 35억\n\n2. **공제금액 결정**:\n   - 일반적인 경우에 해당하므로 9억원을 공제합니다.\n\n3. **공제된 과세표준 계산**:\n   - 과세표준 = (주택 공시가격 합산 - 공제금액) × 공정시장가액비율\n   - 과세표준 = (35억 - 9억) × 60%\n   - 과세표준 = (26억) × 60% = 15.6억\n\n따라서, 이 사용자는 15.6억 원의 과세표준으로 종합부동산세를 계산하게 됩니다. \n\n이제 세금 산출을 위해 종합부동산세 세율을 적용해야 하는데, 실제 세율은 해당 과세표준에 따라 차등적으로 적용됩니다. 2022년 기준, 1세대 1주택자의 경우 1.0%의 세율이 일반적으로 적용되지만 세율은 변동할 수 있습니다.\n\n4. **종합부동산세 예시 계산**:\n   - 세금 = 과세표준 × 세율\n   - 세금 = 15.6억 × 1% = 1560만원\n\n결론적으로, 주택 공시가격이 35억원인 경우의 대략적인 종합부동산세는 약 1560만원입니다. \n(실제 세율은 변동할 수 있으므로, 정확한 계산은 최신 세법에 따라 확인해야 합니다.)', 'query': query}


# %%
calculate_tax_rate(tax_base_state)

# %%
# 노드, 엣지 추가

graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)

# %%
# 엣지 추가

from langgraph.graph import START, END


graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')

graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')


graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)


graph = graph_builder.compile()
