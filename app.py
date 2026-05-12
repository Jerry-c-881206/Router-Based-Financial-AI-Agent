import streamlit as st

from query_understanding import understand_query
from execution_planner import plan_and_execute
from response_generator import render_response

st.set_page_config(
    page_title="理財助理",
    page_icon="📈",
    layout="centered"
)

st.title("📈 AI 理財助理")
st.caption("根據您的個股問題，即時搜尋並摘要重點資訊")

# 對話歷史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示歷史對話
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 使用者輸入
if user_input := st.chat_input("請輸入您的問題，例如：台積電最近的營收狀況？"):
    
    # 顯示使用者訊息
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # v2.0 Router-based flow
    with st.chat_message("assistant"):
        with st.spinner("搜尋中，請稍候..."):
            qu = understand_query(user_input)
            print(f"qu: {qu}")
            planner_result = plan_and_execute(qu, user_question=user_input)
            print(f"planner_result: {planner_result}")
            response = render_response(planner_result, user_question=user_input)
            print(f"response: {response}")
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})