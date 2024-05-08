import re
from time import sleep
from typing import Generator
import streamlit as st
import torch
from generative_model import fix_sentence, predict_suggestions


def stream_data(text: str) -> Generator[str, None, None]:
    """
    This function streams data to the Streamlit app.

    Upon hitting the "Process" button, the function streams the processed text to the Streamlit app.
    """
    for char in text:
        sleep(0.01)
        yield char


def give_suggestions(status: bool = True):
    """
    This function gives token suggestions to the user.

    Upon hitting the "Suggest" button, the function gives token suggestions to the user.
    """
    st.session_state.suggest_token = status


def update_text(processed_text: str):
    """
    This function updates the text in the Streamlit app.

    Upon hitting the "Replace text" button, the function updates the text in the Streamlit app.
    """
    st.session_state.text = processed_text
    give_suggestions()


def main():
    """
    Main function for the Streamlit app.

    All the Streamlit elements are defined here.
    """

    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available(): # ! pytorch-nightly required for using mps
    #     device = "mps"
    else:
        device = "cpu"

    if "site_reloads" not in st.session_state:
        st.session_state.site_reloads = 0
    st.session_state.site_reloads += 1

    if "text" not in st.session_state:
        st.session_state.text = (
            "Talk like Yoda I will. Very wise he was. Strong with the force he was."
        )

    if "suggest_token" not in st.session_state:
        st.session_state.suggest_token = True

    if "last_suggestions" not in st.session_state:
        st.session_state.last_suggestions = None

    st.title("Writing Assistant")

    # The sidebar

    with st.sidebar:
        if device == "cuda":
            st.write("## Transformation Settings")

            task = st.selectbox(
                "Task",
                options=[
                    "paraphrase",
                    "coherent",
                    "simpler",
                    "grammar",
                    "formal",
                    "neutral",
                ],
                on_change=give_suggestions,
                args=(False,),
            )

            decoding_strategy_processing = st.selectbox(
                "Decoding Strategy",
                ["stochastic", "greedy"],
                on_change=give_suggestions,
                args=(False,),
            )

            max_length_processing = st.slider(
                "Max sentence length",
                50,
                500,
                200,
                on_change=give_suggestions,
                args=(False,),
            )

        st.write("## Token Suggestions Settings")

        decoding_strategy_suggestions = st.selectbox(
            "Decoding strategy for suggestions:",
            ["stochastic", "greedy"],
            on_change=give_suggestions,
        )

        num_token_suggestions = st.slider(
            "Max token suggestions (blankspace will be omitted)",
            1,
            10,
            5,
            on_change=give_suggestions,
        )
        token_suggestion_qty = st.slider(
            "Words per suggestion",
            1,
            10,
            1,
            on_change=give_suggestions,
        )

    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        user_input = st.text_area(
            "**Enter your text here:**",
            key="text", # This will fetch the value of key "text" from the dict st.session_state
            on_change=give_suggestions,
            placeholder="Type something...",
        )

        if device == "cuda":
            st.write(f"Apply the transformation: :green[**{task.capitalize()}**]")

            if st.button("Transform"):
                with st.status("Transforming...", state="running") as status:
                    st.write(":orange[**Thinking...**]")
                    processed_text = fix_sentence(
                        task=task,
                        input_text=user_input,
                        decoding_strategy=decoding_strategy_processing,
                        max_length=max_length_processing,
                        device=device,
                        low_memory_setting=True,
                    )
                    st.write(":green[**Transformation completed!**]")
                # st.write(processed_text)
                processed_text = re.sub(
                    r"generated.\s?", "", processed_text, flags=re.I
                )
                if processed_text:
                    status.update(label=":green[**Success!**]", state="complete")
                    st.write_stream(stream_data(processed_text))
                    st.session_state.suggest_token = False
                    st.button(
                        "Replace text",
                        on_click=update_text,
                        args=(processed_text,),
                    )
                else:
                    status.update(
                        label=":orange[**No suggestions!**]", state="complete"
                    )
        else:
            if st.session_state.site_reloads == 1:
                st.error("Use a GPU for more features!")

    with col2:
        st.caption("**Token Suggestions**")
        if st.session_state.suggest_token:
            with st.spinner("Generating token suggestions..."):
                suggestions = predict_suggestions(
                    input_text=user_input,
                    num_token_suggestions=num_token_suggestions,
                    token_suggestion_qty=token_suggestion_qty,
                    decoding_strategy=decoding_strategy_suggestions,
                    device=device,
                    low_memory_setting=True,
                )
                st.session_state.last_suggestions = suggestions

        # put the suggestions in a bullet list
        if st.session_state.last_suggestions:
            for _, suggestion in enumerate(st.session_state.last_suggestions):
                if suggestion == "\n":
                    button_placeholder = "<newline>"
                elif suggestion == " ":
                    button_placeholder = "<space>"
                else:
                    button_placeholder = suggestion
                st.button(
                    rf"{button_placeholder}",
                    on_click=update_text,
                    args=(st.session_state.text + suggestion,),
                )
        else:
            st.warning("No suggestions available")


if __name__ == "__main__":
    main()
