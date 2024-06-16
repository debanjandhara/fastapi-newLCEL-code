from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model="gpt-3.5-turbo-1106")

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | chat

chain.invoke(
    {
        "messages": [
            HumanMessage(
                content="Translate this sentence from English to French: I love programming."
            ),
            AIMessage(content="J'adore la programmation."),
            HumanMessage(content="What did you just say?"),
        ],
    }
)

from langchain_community.chat_message_histories import ChatMessageHistory

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message(
    "Translate this sentence from English to French: I love programming."
)

demo_ephemeral_chat_history.add_ai_message("J'adore la programmation.")

demo_ephemeral_chat_history.messages

demo_ephemeral_chat_history = ChatMessageHistory()

input1 = "Translate this sentence from English to French: I love programming."

demo_ephemeral_chat_history.add_user_message(input1)

response = chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    }
)

demo_ephemeral_chat_history.add_ai_message(response)

input2 = "What did I just ask you?"

demo_ephemeral_chat_history.add_user_message(input2)

chain.invoke(
    {
        "messages": demo_ephemeral_chat_history.messages,
    }
)

# Automatic Chat History 

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat

chain_with_message_history.invoke(
    {"input": "Translate this sentence from English to French: I love programming."},
    {"configurable": {"session_id": "unused"}},
)

# Auto adding stuffs

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("Hey there! I'm Nemo.")
demo_ephemeral_chat_history.add_ai_message("Hello!")
demo_ephemeral_chat_history.add_user_message("How are you today?")
demo_ephemeral_chat_history.add_ai_message("Fine thanks!")

demo_ephemeral_chat_history.messages

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

chain = prompt | chat

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

chain_with_message_history.invoke(
    {"input": "What's my name?"},
    {"configurable": {"session_id": "unused"}},
)

# Trimming Message

from langchain_core.runnables import RunnablePassthrough


def trim_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) <= 2:
        return False

    demo_ephemeral_chat_history.clear()

    for message in stored_messages[-2:]:
        demo_ephemeral_chat_history.add_message(message)

    return True


chain_with_trimming = (
    RunnablePassthrough.assign(messages_trimmed=trim_messages)
    | chain_with_message_history
)

chain_with_trimming.invoke(
    {"input": "Where does P. Sherman live?"},
    {"configurable": {"session_id": "unused"}},
)


# Summary Memory :

demo_ephemeral_chat_history = ChatMessageHistory()

demo_ephemeral_chat_history.add_user_message("Hey there! I'm Nemo.")
demo_ephemeral_chat_history.add_ai_message("Hello!")
demo_ephemeral_chat_history.add_user_message("How are you today?")
demo_ephemeral_chat_history.add_ai_message("Fine thanks!")

demo_ephemeral_chat_history.messages

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability. The provided chat history includes facts about the user you are speaking with.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ]
)

chain = prompt | chat

chain_with_message_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: demo_ephemeral_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def summarize_messages(chain_input):
    stored_messages = demo_ephemeral_chat_history.messages
    if len(stored_messages) == 0:
        return False
    summarization_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                "Distill the above chat messages into a single summary message. Include as many specific details as you can.",
            ),
        ]
    )
    summarization_chain = summarization_prompt | chat

    summary_message = summarization_chain.invoke({"chat_history": stored_messages})

    demo_ephemeral_chat_history.clear()

    demo_ephemeral_chat_history.add_message(summary_message)

    return True


chain_with_summarization = (
    RunnablePassthrough.assign(messages_summarized=summarize_messages)
    | chain_with_message_history
)

chain_with_summarization.invoke(
    {"input": "What did I say my name was?"},
    {"configurable": {"session_id": "unused"}},
)

