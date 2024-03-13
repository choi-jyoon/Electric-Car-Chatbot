import streamlit as st
from dotenv import load_dotenv
import os 
import pandas as pd

from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.schema import ChatMessage
import textwrap

MAX_INPUT_TOKENS = 100  # 입력 메시지의 최대 토큰 수

load_dotenv()

MODEL = "gpt-3.5-turbo"
API_KEY = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(api_key=API_KEY)

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

want_to = """너는 아래 내용을 기반으로 질의응답을 하는 로봇이야.
content
{}
"""

# txt 파일 읽어서 프롬프트 내용 확인
with open('충전소현황.txt', 'r', encoding='utf-8') as f:
    content = f.read()


st.header("전기차 충전소 현황 정보")
st.info("전국 전기차 충전소 정보를 알려주는 챗봇입니다.")
# st.error("교육 커리큘럼 내용이 적용되어 있습니다.")

with st.sidebar:
    st.header('전기차 충전소 Chatbot', divider = 'rainbow')
    #st.write('This chatbot is created using the open-source Llama 2 LLM model from Meta.')
    def dx_chat():
        st.session_state.messages = [ChatMessage(role="assistant", content="안녕하세요! 전기차 충전소 정보입니다. 어떤 내용이 궁금하신가요?")]

    # 채팅 내용 지우기
    def clear_chat():
        st.session_state.messages = [ChatMessage(role="assistant", content="안녕하세요! 전기차 충전소 정보 챗봇입니다. 어떤 내용이 궁금하신가요?")]
    st.sidebar.button('Clear Chat', on_click=clear_chat, type='primary')


if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="안녕하세요! 전기차 충전소 정보 챗봇입니다. 어떤 내용이 궁금하신가요?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    if not API_KEY:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = ChatOpenAI(openai_api_key=API_KEY, streaming=True, callbacks=[stream_handler], model_name=MODEL)

    # 사용자 입력 메시지를 요약하여 최대 토큰 수를 초과하지 않도록 함
    if len(prompt.split()) > MAX_INPUT_TOKENS:
        prompt = ' '.join(prompt.split()[:MAX_INPUT_TOKENS])
        prompt = textwrap.shorten(prompt, width=MAX_INPUT_TOKENS)

    # 요약된 입력 메시지를 사용하여 모델에 전달
    response = llm([ ChatMessage(role="system", content=want_to.format(content))]+st.session_state.messages)
    st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))