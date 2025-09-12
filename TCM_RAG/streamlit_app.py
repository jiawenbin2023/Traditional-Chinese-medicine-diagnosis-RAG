# streamlit_app.py
import streamlit as st
import requests
import json
import time # 导入time模块
from typing import Dict, Any, List

# 从config获取API地址
from config.settings import Settings

app_settings = Settings()
FASTAPI_URL = f"http://{app_settings.app.api_host}:{app_settings.app.api_port}"

st.set_page_config(page_title="企业级中医RAG问答", layout="wide")

st.title("🤖 企业级中医RAG问答系统")

# --- Sidebar ---
st.sidebar.header("系统设置")

# ==
# 健壮的API状态检查 (新增重试和详细错误处理)
# ==
max_retries = 5  # 最大重试次数
retry_delay = 1  # 每次重试间隔秒数

api_status_message = "未知"
for attempt in range(max_retries):
    try:
        health_response = requests.get(f"{FASTAPI_URL}/health", timeout=5) # 增加超时时间
        if health_response.status_code == 200:
            try:
                health_data = health_response.json()
                api_status_message = f"✅ 运行中 ({health_data.get('status', 'Healthy')})"
            except json.JSONDecodeError:
                # 如果是200但不是JSON，说明API行为异常
                api_status_message = f"❌ API返回非JSON格式 (Status: {health_response.status_code})"
            break  # 成功获取状态并解析，跳出重试循环
        else:
            api_status_message = f"❌ 错误 ({health_response.status_code})"
            # 对于非200错误，如果是503或502，可能值得重试
            if health_response.status_code in [502, 503]:
                st.sidebar.warning(f"API返回 {health_response.status_code} (尝试 {attempt + 1}/{max_retries})，等待重试...")
                time.sleep(retry_delay)
                continue # 继续下一次重试
            break # 其他错误则不重试
    except requests.exceptions.ConnectionError:
        api_status_message = f"❌ 未连接到API服务 (尝试 {attempt + 1}/{max_retries})，等待重试..."
        st.sidebar.warning(api_status_message)
        time.sleep(retry_delay)
    except requests.exceptions.Timeout:
        api_status_message = f"❌ API连接超时 (尝试 {attempt + 1}/{max_retries})，等待重试..."
        st.sidebar.warning(api_status_message)
        time.sleep(retry_delay)
    except Exception as e:
        api_status_message = f"❌ API连接发生未知错误: {e}"
        break # 发生未知错误，不重试
else:
    # 如果所有重试都失败
    if api_status_message == "未知": # 如果从未更新过，则提供一个通用错误
        api_status_message = "❌ 无法连接到API服务 (多次尝试后失败)"

st.sidebar.write(f"**API状态:** {api_status_message}") # 使用健壮检查后的状态信息

if st.sidebar.button("清空Redis缓存"):
    try:
        response = requests.post(f"{FASTAPI_URL}/clear_cache", timeout=10) # 增加超时时间
        if response.status_code == 200:
            st.sidebar.success("Redis缓存已清空！")
        else:
            # 尝试解析错误详情，如果不是JSON则显示原始文本
            try:
                error_detail = response.json().get('detail', response.text)
            except json.JSONDecodeError:
                error_detail = response.text
            st.sidebar.error(f"清空缓存失败: {error_detail}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error("无法连接到API服务。")
    except requests.exceptions.Timeout:
        st.sidebar.error("API连接超时。")
    except Exception as e:
        st.sidebar.error(f"清空缓存发生未知错误: {e}")

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("请输入你的问题..."):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # 调用FastAPI后端
            api_response = requests.post(
                f"{FASTAPI_URL}/query",
                json={"query": prompt, "user_id": "streamlit_user", "include_debug": True},
                timeout=60 # 增加查询超时时间，因为LLM推理可能比较慢
            )
            api_response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)

            rag_result = api_response.json() # 现在这里应该能成功解析了

            answer = rag_result.get("answer", "未能获取答案。")
            sources = rag_result.get("sources", [])
            total_time = rag_result.get("total_time", 0)
            confidence = rag_result.get("confidence", 0)
            cache_hit = rag_result.get("cache_hit", False)
            method_used = rag_result.get("method_used", "hybrid")

            full_response += f"💡 **答案:**\n{answer}\n\n"
            full_response += f"📊 **置信度:** {confidence:.2%}\n"
            full_response += f"⚡ **缓存命中:** {'是' if cache_hit else '否'} (召回方式: {method_used})\n"
            full_response += f"⏱ **总耗时:** {total_time:.2f} 秒\n\n"

            if sources:
                full_response += "**📚 参考资料来源:**\n"
                for i, src in enumerate(sources, 1):
                  file_name = src.get('file_name', '未知文件')
                  score = src.get('score', 0)
                  preview = src.get('preview', '无预览')
                  full_response += f" {i}. 文件: `{file_name}` | 分数: {score:.4f} | 预览: {preview}\n"
            else:
                full_response += "**📚 参考资料来源:** 无\n"

            message_placeholder.markdown(full_response)

        except requests.exceptions.ConnectionError:
            full_response = "❌ 无法连接到RAG服务API，请确保后端已启动并监听正确端口。"
            message_placeholder.error(full_response)
        except requests.exceptions.Timeout:
            full_response = "❌ API请求超时，可能是后端处理时间过长或网络问题。"
            message_placeholder.error(full_response)
        except requests.exceptions.HTTPError as e:
            # 尝试解析错误详情，如果不是JSON则显示原始文本
            try:
                error_detail = e.response.json().get('detail', e.response.text)
            except json.JSONDecodeError:
                error_detail = e.response.text
            full_response = f"❌ API请求失败: {error_detail}"
            message_placeholder.error(full_response)
        except Exception as e:
            full_response = f"❌ 发生未知错误: {e}"
            message_placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
