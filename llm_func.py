from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from langchain_community.chat_message_histories import ChatMessageHistory

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain_pinecone import PineconeVectorStore

from example import answer_examples

# common variables
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_retriever():
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")
    index_name = "incometax-markdown"
    db = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    return db.as_retriever(search_kwargs={'k':4})


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    context_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history,"
        "formulate a standalone question which can be understood without the chat history."
        "Do NOT answer the question, just reformulate it if needed and otherwise it as is."
    )
    context_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", context_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_retriever = create_history_aware_retriever(llm, retriever, context_q_prompt)

    return history_retriever


def get_llm(model: str = "gpt-4o"):
    llm = ChatOpenAI(model=model)
    return llm

def get_dict_chain():
    llm = get_llm()
    query_dict = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해 주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런경우에는 질문만 반환해주세요.
        사전: {query_dict}

        질문: {{question}}
        """)

    dict_chain = prompt | llm | StrOutputParser()
    return dict_chain


def get_rag_chain():
    llm = get_llm()

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}")
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    system_prompt = (
        "당신은 한국 소득세법 전문가입니다. 사용자의 소득세법에 대한 질문에 답해주세요"
        "아래에 제공된 문서를 활용해서 답해주시고"
        "답을 알 수 없다면 모른다고 답변해 주세요."
        "답변은 소득세법 (??조)에 따르면과 같이 근거를 먼저 제시하면서 답변해주세요"
        "2~3문장정도의 짧은 답변이면 됩니다."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,                                        # Few shot 예시를 시스템 메세지와 질문 사이에 집어넣어서 과거 이렇게 대화했던 것처럼 속임 . 없으면 zero-shot
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_retriever = get_history_retriever()
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_retriever, qa_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick("answer")

    return conversational_rag_chain


def get_ai_msg(user_msg: str = None):
    dict_chain = get_dict_chain()
    rag_chain = get_rag_chain()
    tax_chain = {"input": dict_chain} | rag_chain
    ai_response = tax_chain.stream(
        {
            "question": user_msg
         },
        config={"configurable": {"session_id": "testid"}}
    )

    return ai_response