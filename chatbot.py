"""
Gemini 2.0 Flash 聊天機器人
使用 langchain-google-genai 串接，具備對話記憶與歷史紀錄持久化功能。
"""

import json
import os
import atexit
from datetime import datetime

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# ── 載入環境變數 ──────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("請在 .env 檔案中設定 GEMINI_API_KEY")

# ── 初始化模型 ────────────────────────────────────────────────
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=api_key,
    temperature=0.7,
)

# ── 對話歷史管理 ──────────────────────────────────────────────
store: dict[str, InMemoryChatMessageHistory] = {}
SESSION_ID = "default"

# 用來記錄帶時間戳的對話紀錄
conversation_log: list[dict] = []


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """取得或建立指定 session 的對話歷史。"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


chain_with_history = RunnableWithMessageHistory(
    llm,
    get_session_history,
)


# ── 對話紀錄持久化 ────────────────────────────────────────────
def save_conversation():
    """將對話紀錄存成 JSON 檔案。"""
    if not conversation_log:
        return

    filename = datetime.now().strftime("chat_%Y%m%d_%H%M%S.json")
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=2)

    print(f"\n💾 對話紀錄已儲存至：{filepath}")


# 註冊程式結束時自動儲存
atexit.register(save_conversation)


# ── 主程式 ────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("🤖 Gemini 2.0 Flash 聊天機器人")
    print("=" * 50)
    print("輸入訊息開始聊天，輸入 'exit' 結束對話。\n")

    while True:
        try:
            user_input = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n👋 再見！")
            break

        # 記錄使用者訊息
        conversation_log.append({
            "timestamp": datetime.now().isoformat(),
            "role": "user",
            "content": user_input,
        })

        # 呼叫模型取得回應
        try:
            response = chain_with_history.invoke(
                [HumanMessage(content=user_input)],
                config={"configurable": {"session_id": SESSION_ID}},
            )
            ai_content = response.content

            # 記錄 AI 回應
            conversation_log.append({
                "timestamp": datetime.now().isoformat(),
                "role": "ai",
                "content": ai_content,
            })

            print(f"\nAI：{ai_content}\n")

        except Exception as e:
            print(f"\n❌ 發生錯誤：{e}\n")

    # 儲存對話紀錄（atexit 也會觸發，但這裡提前呼叫以確保正常退出時也存檔）
    save_conversation()


if __name__ == "__main__":
    main()
