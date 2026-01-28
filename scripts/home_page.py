import gradio as gr
from run_agent import run


def on_user_submit(history, user_text):
    """只负责把用户消息追加到 history，并立刻返回（UI 立即刷新）"""
    user_text = (user_text or "").strip()
    history = history or []

    if not user_text:
        return history, ""

    history.append({"role": "user", "content": user_text})
    return history, ""


def on_generate_answer(history):
    """在已有 history 基础上生成回答（慢操作放这里）"""
    history = history or []
    if not history:
        return history

    # 找到最后一条用户消息（更稳，不假设最后一条一定是 user）
    last_user = None
    for m in reversed(history):
        if m.get("role") == "user":
            last_user = m.get("content")
            break

    if not last_user:
        history.append({"role": "assistant", "content": "我没有收到问题，请再输入一次。"})
        return history

    try:
        answer = run(last_user)
    except Exception as e:
        answer = f"运行出错：{type(e).__name__}: {e}"

    history.append({"role": "assistant", "content": answer})
    return history


with gr.Blocks(title="红楼梦 RAG 智能助手") as demo:
    gr.Markdown("## 《红楼梦》RAG 智能助手（Hybrid 检索 + 邻近扩展 + LLM）")

    chatbot = gr.Chatbot(label="对话", height=720)

    user_input = gr.Textbox(
        label="输入问题",
        placeholder="例如：谁写了柳絮词？",
        lines=2,
    )

    send_btn = gr.Button("发送", variant="primary")
    clear_btn = gr.Button("清空对话")

    # 点击发送：先立即显示用户消息 -> 再生成回答
    send_btn.click(
        on_user_submit,
        inputs=[chatbot, user_input],
        outputs=[chatbot, user_input],
    ).then(
        on_generate_answer,
        inputs=[chatbot],
        outputs=[chatbot],
    )

    # 回车提交：同样两段式
    user_input.submit(
        on_user_submit,
        inputs=[chatbot, user_input],
        outputs=[chatbot, user_input],
    ).then(
        on_generate_answer,
        inputs=[chatbot],
        outputs=[chatbot],
    )

    clear_btn.click(lambda: [], inputs=None, outputs=[chatbot])


if __name__ == "__main__":
    demo.launch()
