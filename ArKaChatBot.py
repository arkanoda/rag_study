from dotenv import load_dotenv

from llm_func import get_ai_msg

import streamlit as st

load_dotenv()

# page config
st.set_page_config(page_title="ArKa ChatBot", page_icon=":robot:")

# 제목, 설명
st.title("ArKaNoD's ChatBot 🤖")
st.caption("질문에 답변해드립니다.")



# 채팅 입력창, placeholder는 빈칸일 때 표시되는 내용
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 기존 메세지 리스트에 있는 것들 화면에 그려주는 코드 > 자동으로 변경사항만 update해줌
for msg in st.session_state.message_list:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

if user_question := st.chat_input(placeholder="궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    # 세션에 채팅내용 추가 저장하는 코드
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변 생성 중"):
        ai_response = get_ai_msg(user_question)
        with st.chat_message("ai"):
            ai_msg = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_msg})
            # 세션에 채팅내용 추가 저장하는 코드

