from openai import OpenAI
from context import TWIN_SYSTEM_PROMPT
from tools import handle_tool_calls, tools
from styles import CSS, JS, EXAMPLES
from dotenv import load_dotenv
import gradio as gr

load_dotenv(override = True)
openai = OpenAI()
system = {
    "role": "system", "content": TWIN_SYSTEM_PROMPT
}


def chat(message, history):
    messages = {"role": "system", "content": TWIN_SYSTEM_PROMPT} + history + {"role": "user", "content": message}
    response = openai.chat.completions.create(
        model = "gpt-5.4-mini", messages = messages
    )
    while response.choices[0].finish_reason == "tool_calls":
        tool_calls = response.choices[0].message
        results = handle_tool_calls(tool_calls)
        messages.append(message)
        messages.extend(results)
        response = openai.chat.completions.create(
            model = "gpt-5.4-mini", messages = messages, tools = tools
        )
    return response.choices[0].message.content

if __name__ == "__main__":
    gr.ChatInterface(
        chat,
        examples=EXAMPLES,
        title="Digital Twin",
        description="Talk to my AI twin about my career",
        chatbot=gr.Chatbot(show_label=False),
    ).launch(css=CSS, js=JS, theme=gr.themes.Base())
