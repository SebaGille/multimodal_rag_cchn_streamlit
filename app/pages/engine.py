from __future__ import annotations

from copy import deepcopy

import streamlit as st

from rag_state import (
    DEFAULT_SETTINGS,
    get_rag_settings,
    get_run_state,
    update_rag_settings,
)


def main() -> None:
    st.set_page_config(page_title="Engine", page_icon="")

    settings = get_rag_settings()
    run_state = get_run_state()

    st.title("Engine")
    st.write(
        "Adjust the retrieval and generation settings that power the chatbot. "
        "Changes here apply immediately to the next question you ask."
    )

    with st.form("rag_settings_form"):
        vectorstore_path = st.text_input(
            "Vectorstore directory",
            value=settings["vectorstore_path"],
            help="Location of the FAISS index used for retrieval.",
        )
        embedding_model = st.text_input(
            "Embedding model",
            value=settings["embedding_model"],
            help="Model that embeds passages and questions.",
        )
        chat_model = st.text_input(
            "Chat model",
            value=settings["chat_model"],
            help="Model that drafts the final answer.",
        )
        top_k = st.slider(
            "Number of passages (k)",
            min_value=1,
            max_value=8,
            value=int(settings["top_k"]),
            help="How many passages to retrieve before answering.",
        )
        prompt_template = st.text_area(
            "Prompt template",
            value=settings["prompt_template"],
            height=220,
            help="Guidance provided to the language model.",
        )
        submitted = st.form_submit_button("Save settings")

        if submitted:
            update_rag_settings(
                {
                    "vectorstore_path": vectorstore_path.strip() or settings["vectorstore_path"],
                    "embedding_model": embedding_model.strip() or settings["embedding_model"],
                    "chat_model": chat_model.strip() or settings["chat_model"],
                    "top_k": int(top_k),
                    "prompt_template": prompt_template.strip() or settings["prompt_template"],
                }
            )
            st.success("Settings updated. Ask a question on the Chatbot page to try them out.")

    if st.button("Reset settings to defaults"):
        update_rag_settings(deepcopy(DEFAULT_SETTINGS))
        st.success("Settings reset. Re-run your question from the Chatbot page.")

    st.divider()
    st.subheader("Latest run details")
    if run_state.get("question"):
        st.markdown(f"- Last question: `{run_state['question']}`")
        st.markdown(f"- Passages retrieved: {run_state.get('retrieved_count', 0)}")
        last_settings = run_state.get("last_run_settings", {})
        if last_settings:
            st.markdown(
                f"- Embedding model used: `{last_settings.get('embedding_model', 'NA')}`"
            )
            st.markdown(f"- Chat model used: `{last_settings.get('chat_model', 'NA')}`")
    else:
        st.info("No questions asked yet. Start on the Chatbot page to populate this section.")

    if run_state.get("error"):
        st.error(f"Most recent error: {run_state['error']}")


if __name__ == "__main__":
    main()

