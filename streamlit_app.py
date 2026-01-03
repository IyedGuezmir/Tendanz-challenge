import streamlit as st
from app.src.generation import rag_fusion_chain
from langchain.callbacks.base import BaseCallbackHandler


class StreamlitCallback(BaseCallbackHandler):
    def __init__(self, container, status_container):
        self.container = container
        self.status_container = status_container
        self.text = ""
        self.started = False

    def on_llm_new_token(self, token: str, **kwargs):
        # Clear status message once first token arrives
        if not self.started:
            self.status_container.empty()
            self.started = True

        self.text += token
        self.container.markdown(self.text)


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Tendanz Insurance Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.markdown("## Tendanz Insurance Chatbot")
st.markdown(
    "Ask legal questions and get fast, accurate answers based on **Auto MMA insurance policy documents**."
)

user_question = st.text_area(
    "Enter your question:",
    placeholder="Type your legal question here..."
)

if st.button("Ask"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        # Placeholders
        status_container = st.empty()
        answer_container = st.empty()

        # Show immediate feedback
        status_container.markdown("🔍 **Searching relevant legal documents…**")

        # Retrieve documents
        docs = rag_fusion_chain.retrieve(user_question)
        context = "\n\n".join(doc.page_content for doc in docs)

        status_container.markdown("🧠 **Analyzing legal context and generating answer…**")

        # Prepare callback
        callback = StreamlitCallback(
            container=answer_container,
            status_container=status_container
        )

        prompt = rag_fusion_chain.prompt_answer.format_prompt(
            context=context,
            question=user_question
        )

        # Stream answer
        rag_fusion_chain.llm.generate_prompt(
            [prompt],
            callbacks=[callback]
        )
