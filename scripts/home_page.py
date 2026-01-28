import gradio as gr
from run_agent import run


def chat(history, user_text):
    user_text = (user_text or "").strip()
    if not user_text:
        return history, ""

    history = history or []
    history.append({"role": "user", "content": user_text})

    try:
        answer = run(user_text)
    except Exception as e:
        answer = f"运行出错：{type(e).__name__}: {e}"

    history.append({"role": "assistant", "content": answer})
    return history, ""


with gr.Blocks(title="红楼梦 RAG 智能助手") as demo:
    gr.Markdown("## 《红楼梦》RAG 智能助手（Hybrid 检索 + 邻近扩展 + LLM）")

    chatbot = gr.Chatbot(
        label="对话",
        height=520,
    )

    user_input = gr.Textbox(
        label="输入问题",
        placeholder="例如：谁写了柳絮词？",
        lines=2,
    )

    send_btn = gr.Button("发送", variant="primary")

    send_btn.click(
        chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, user_input],
    )

    user_input.submit(
        chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, user_input],
    )


if __name__ == "__main__":
    demo.launch()
