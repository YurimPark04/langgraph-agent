from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langgraph.graph import START, END
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph
from langchain_ollama import ChatOllama
from langsmith import Client
from typing import Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate



# 이제 임베딩 생성에는 OPENAI_API_KEY도 필요 없고 API 비용 발생 X
embedding_function = OllamaEmbeddings(
    model="bge-m3",
    base_url="http://localhost:11434"
)

# 이미 생성된 db 에서 조회하는 코드 (Chroma 클래스를 바로 활용한다  )
vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name='income_tax_collection',
    persist_directory='../income_tax_collection'   # 로컬에 설정한 경로로 데이터가 남음 (DB의 역할을 한다)
)

retriever = vector_store.as_retriever(search_kwargs={'k': 3})

# %%

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str
    
graph_builder = StateGraph(AgentState)

# %%
# retrieve 노드는 사용자의 질문을 받아 벡터 스토어에서 추출한 데이터를 반환한다

def retrieve(state: AgentState):
    query = state["query"]  # state 에서 query 를 꺼내온다.
    
    docs = retriever.invoke(query)  # 검색한 문서

    # 검색한 문서를 state의 context 에 넣어준다 (docs를 담아준다)
    return {'context': docs}  # 여기서 키값 (context)은 앞서서 state를 선언한 이름과 동일해야 함

# %%

# LLM 선언

# 추후에 num_ctx, num_predict 등 다른 인자 제외하고 테스트 해볼 것
llm = ChatOllama(
    model='exaone3.5:7.8b',
    # temperature=0,
    # reasoning=False,  # 긴 내부 추론을 끔
    # num_ctx=8192,  # 입력과 출력을 합친 컨텍스트 한도를 늘림
)

# %%


client = Client()

# %%
generate_prompt = client.pull_prompt(
    "rlm/rag-prompt",
    dangerously_pull_public_prompt=True,
    )

# 디버깅 이후
generate_llm = ChatOllama(
    model='exaone3.5:7.8b', 
    num_predict=512,  # 생성할 최대 토큰 수
)

def generate(state:AgentState):
    context = state['context']
    query = state['query']

    rag_chain = generate_prompt | generate_llm  # 디버깅 이후

    # Langsmith 공식문서에 보면, question과 context를 인자로 받는다
    response = rag_chain.invoke({'question': query, 'context': context})  

    # response 를 return
    # return {'answer': response}

    return {'answer': response.content}   # 디버깅 후

# 프롬프트를 기반으로 LLM을 호출한다.

# %%
# Langsmith의 프롬프트 활용 - 노션 참고

doc_relevance_prompt = client.pull_prompt(
    "langchain-ai/rag-document-relevance",
    dangerously_pull_public_prompt=True,
)

"""
check_doc_relevance 노드의 반환값을 아래와 같이 로 하게될 경우, 에러가 난다

    ```
    return 'END'
    ```

    END는 노드가 아니라서 Literal로 넘겨줘도 작동을 하지 않는다. -> 엣지 추가 시점에서 작업을 진행해줘야 함
    'relevant' 와 'irrelevant' 노드를 return 하는 걸로 대체
"""

def check_doc_relevance(state:AgentState) -> Literal["relevant", "irrelevant"]:
    context = state['context']
    query = state['query']


    doc_relevance_chain = doc_relevance_prompt | llm

    response = doc_relevance_chain.invoke({'question': query, 'documents': context})  
    print(f'response== {response}')
  
    if response['Score'] == 1:
        return 'relevant'
    
    return 'irrelevant'   

dictionary = ['사람과 관련된 표현 -> 거주자']  # dictionary 선언

rewrite_prompt = PromptTemplate.from_template(
    f""" 사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
    사전: {dictionary}
    질문: {{query}}
    """
)

def rewrite(state:AgentState):
    query = state['query']

    rewrite_chain = rewrite_prompt | llm | StrOutputParser()   # 체이닝

    response = rewrite_chain.invoke({'query': query})  

    return {'query': response}


# 페르소나를 제공
# hallucination 이 어떤 건지에 대한 정보 제공 
hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on documents or not,
Given documents, which are excerpts from income tax law, and a student's answer;
If the student's answer is based on documents, respond with "not hallucinated",
If the student's answer is not based on documents, respond with "hallucinated".

documents: {documents}
student_answer: {student_answer}

""")


hallucination_llm = ChatOllama(
    model='exaone3.5:7.8b', 
    temperature=0
)

def check_hallucination(state:AgentState) -> Literal['hallucinated', 'not hallucinated']:

    answer = state['answer']   # 답변을 봐야함
    context = state['context']   
    context = [doc.page_content for doc in context]  # 디버깅 이후
    print(f'answer === {answer}')

    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()  # 디버깅 후
    response = hallucination_chain.invoke({'student_answer': answer, 'documents':context})   # 공식문서: student_answer, documents 를 대입해야 함

    print(f'hallucination response === {response}')

    return response


helpfulness_prompt = client.pull_prompt(
    "langchain-ai/rag-answer-helpfulness",
    dangerously_pull_public_prompt=True
    )

def check_helpfulness_grader(state: AgentState):
    # state에서 질문과 답변을 추출
    query = state['query'] 
    answer = state['answer']
    
    # 답변의 유용성을 평가하기 위한 체인을 생성합니다
    helpfulness_chain = helpfulness_prompt | llm

    response = helpfulness_chain.invoke({'question':query, 'student_answer':answer})
    print(f"response == {response}")

    if response['Score'] == 1:
        return 'helpful'

    return 'unhelpful'


def check_helpfulness(state: AgentState):   # 디버깅
    return state



graph_builder.add_node('retrieve', retrieve)
graph_builder.add_node('generate', generate)
graph_builder.add_node('rewrite', rewrite)
graph_builder.add_node('check_helpfulness', check_helpfulness)

# %%


graph_builder.add_edge(START, 'retrieve')


graph_builder.add_conditional_edges(
    'retrieve',
    check_doc_relevance,
    {
        'relevant': 'generate',
        'irrelevant': END
    }
)

graph_builder.add_conditional_edges(
    'generate',
    check_hallucination,
    {
        'not hallucinated': 'check_helpfulness',
        'hallucinated': 'generate'   # hallucination 이면 답변을 다시 생성해야 함
    }
)


graph_builder.add_conditional_edges(
    'check_helpfulness',
    # check_helpfulness,
    check_helpfulness_grader,
    {
        'helpful': END,
        'unhelpful': 'rewrite'
    }
)

graph_builder.add_edge('rewrite', 'retrieve')

# %%
graph = graph_builder.compile()
