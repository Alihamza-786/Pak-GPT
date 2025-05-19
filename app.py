import chainlit as cl
import uuid
from langchain_ollama import ChatOllama
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate
)
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. LLM setup with streaming enabled
llm = ChatOllama(model="llama3.1", temperature=0.5, streaming=True)

# 2. System Prompt
system_prompt = (
    "You are an AI assistant called 'PakGPT' that only helps answer questions about Pakistan. "
    "If the user asks about any other country or unrelated topic, respond with: "
    "'I don't know, I am PakGPT, please ask me about Pakistan only.'"
)

# 3. Prompt Template
prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{query}"),
])

# 4. Pipeline
pipeline = prompt_template | llm

# 5. Memory setup per session
chat_map = {}
def get_chat_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_map:
        chat_map[session_id] = InMemoryChatMessageHistory()
    return chat_map[session_id]

# 6. Runnable with memory
pipeline_with_history = RunnableWithMessageHistory(
    pipeline,
    get_session_history=get_chat_history,
    input_messages_key="query",
    history_messages_key="history"
)

# Chainlit app entry point
@cl.on_chat_start
async def start():
    await cl.Message(content="PakGPT is ready. Ask me anything about Pakistan.").send()

@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        cl.user_session.set("session_id", session_id)

    user_text = message.content

    # 1. Send an empty message first (placeholder)
    msg = cl.Message(content="")
    await msg.send()

    # 2. Stream and update the message in real-time
    async for chunk in pipeline_with_history.astream(
        {"query": user_text},
        config={"configurable": {"session_id": session_id}}
    ):
        msg.content += chunk.content
        await msg.update()



