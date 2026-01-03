import streamlit as st
from app.src.generation import rag_fusion_chain
from langchain.callbacks.base import BaseCallbackHandler

class StreamlitCallback(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Tendanz Insurance Chatbot", page_icon="🤖", layout="wide")
st.markdown("## Tendanz Insurance Chatbot")
st.markdown("Ask legal questions and get fast and accurate answers based on Auto MMA insurance policy documents.")

user_question = st.text_area("Enter your question:", placeholder="Type your legal question here...")
if st.button("Ask"):
    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        container = st.empty()
        callback = StreamlitCallback(container)

        # Retrieve top documents for context
        docs = rag_fusion_chain.retrieve(user_question)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = rag_fusion_chain.prompt_answer.format_prompt(
            context=context,
            question=user_question
        )

        # Stream the answer
        rag_fusion_chain.llm.generate_prompt(
            [prompt],  # <--- pass prompt object
            callbacks=[callback]  # <--- streaming callback goes here
        )