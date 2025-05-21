import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from datetime import datetime
import os

# 设置页面配置
st.set_page_config(
    page_title="本地聊天机器人",
    page_icon="🤖",
    layout="wide"
)

# 初始化session state
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate."}
    ]

if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}

# 初始化模型参数
if "temperature" not in st.session_state:
    st.session_state["temperature"] = 0.7
if "max_tokens" not in st.session_state:
    st.session_state["max_tokens"] = 256
if "top_p" not in st.session_state:
    st.session_state["top_p"] = 0.95
if "top_k" not in st.session_state:
    st.session_state["top_k"] = 50

# 创建保存聊天记录的目录
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

def save_chat_history(chat_id, messages):
    """保存聊天记录到文件"""
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)

def load_chat_history(chat_id):
    """从文件加载聊天记录"""
    file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def get_chat_list():
    """获取所有聊天记录文件列表"""
    chats = []
    for file in os.listdir(CHAT_HISTORY_DIR):
        if file.endswith(".json"):
            chat_id = file[:-5]  # 移除.json后缀
            file_path = os.path.join(CHAT_HISTORY_DIR, file)
            timestamp = os.path.getmtime(file_path)
            chats.append({
                "id": chat_id,
                "timestamp": timestamp,
                "date": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            })
    return sorted(chats, key=lambda x: x["timestamp"], reverse=True)

# 标题
st.title("💬 TinyLlama 聊天机器人")
st.caption("TinyLlama-1.1B-Chat-v1.0")

# 侧边栏
with st.sidebar:
    st.header("模型信息")
    st.markdown("""
    - 模型：TinyLlama-1.1B-Chat
    - 版本：v1.0
    - 类型：本地部署
    """)
    
    st.header("参数设置")
    # 温度滑块
    temperature = st.slider(
        "温度 (Temperature)",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.1,
        help="控制输出的随机性。较低的值使输出更确定，较高的值使输出更随机。"
    )
    st.session_state["temperature"] = temperature
    
    # 最大生成长度滑块
    max_tokens = st.slider(
        "最大生成长度 (Max Tokens)",
        min_value=64,
        max_value=512,
        value=st.session_state["max_tokens"],
        step=64,
        help="控制生成回复的最大长度。"
    )
    st.session_state["max_tokens"] = max_tokens
    
    # Top-p 滑块
    top_p = st.slider(
        "Top-p",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state["top_p"],
        step=0.05,
        help="控制输出的多样性。较低的值使输出更集中，较高的值使输出更多样。"
    )
    st.session_state["top_p"] = top_p
    
    # Top-k 滑块
    top_k = st.slider(
        "Top-k",
        min_value=1,
        max_value=100,
        value=st.session_state["top_k"],
        step=1,
        help="控制每次生成时考虑的候选词数量。"
    )
    st.session_state["top_k"] = top_k
    
    st.header("聊天管理")
    
    # 新建聊天按钮
    if st.button("新建聊天"):
        chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state["current_chat"] = chat_id
        st.session_state["messages"] = [
            {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate."}
        ]
        st.rerun()
    
    # 显示聊天历史列表
    st.subheader("历史聊天")
    chats = get_chat_list()
    for chat in chats:
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button(f"📝 {chat['date']}", key=f"chat_{chat['id']}"):
                st.session_state["current_chat"] = chat["id"]
                loaded_messages = load_chat_history(chat["id"])
                if loaded_messages:
                    st.session_state["messages"] = loaded_messages
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{chat['id']}"):
                file_path = os.path.join(CHAT_HISTORY_DIR, f"{chat['id']}.json")
                if os.path.exists(file_path):
                    os.remove(file_path)
                if st.session_state["current_chat"] == chat["id"]:
                    st.session_state["current_chat"] = None
                    st.session_state["messages"] = [
                        {"role": "system", "content": "You are a friendly chatbot who always responds in the style of a pirate."}
                    ]
                st.rerun()

# 初始化模型和tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    return tokenizer, model

# 加载模型
try:
    tokenizer, model = load_model()
except Exception as e:
    st.error(f"模型加载失败: {str(e)}")
    st.stop()

# 展示历史消息
for msg in st.session_state.messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入你的问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # 构建对话模板
    prompt_text = tokenizer.apply_chat_template(
        st.session_state.messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            outputs = model.generate(
                **inputs,
                max_new_tokens=st.session_state["max_tokens"],
                do_sample=True,
                temperature=st.session_state["temperature"],
                top_k=st.session_state["top_k"],
                top_p=st.session_state["top_p"]
            )
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            st.write(response)
            st.session_state["messages"].append({"role": "assistant", "content": response})
            
            # 保存聊天记录
            if st.session_state["current_chat"]:
                save_chat_history(st.session_state["current_chat"], st.session_state["messages"])
            else:
                chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["current_chat"] = chat_id
                save_chat_history(chat_id, st.session_state["messages"]) 