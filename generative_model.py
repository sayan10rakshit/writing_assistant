import transformers
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    T5ForConditionalGeneration,
)
import streamlit as st
import torch
import re


def fix_sentence(
    task: str,
    input_text: str,
    decoding_strategy: str = "stochastic",
    max_length: int = 200,
    device: str = "cpu",
    low_memory_setting=True,
) -> str:
    """
    This function generates text based on the task and input text provided.

    The function uses the Grammarly Coedit model to generate text based on the task and input text
    provided. The model is fine-tuned on the Coedit dataset, which consists of text samples
    annotated with various tasks such as grammar correction, coherence improvement, simplification,
    paraphrasing, formalization, and neutralization. The model uses the T5 architecture and is
    capable of generating text based on the input text and the task provided. The function allows
    for different decoding strategies such as nucleus sampling and greedy decoding. The generated
    text can be used to improve the grammar, coherence, simplicity, paraphrasing, formality, or
    neutrality of the input text.


    Args:
        task (str): The task to be performed by the model. Options are: "grammar", "coherent",
        "simpler", "paraphrase", "formal", "neutral". Defaults to "grammar".

        input_text (str): The input text to be processed by the model

        decoding_strategy (str, optional): The decoding strategy to be used by the model.
        Options are: "stochastic" or "s" for nucleus sampling, "greedy" or "g" for greedy decoding.
        Defaults to "stochastic".

        max_length (int, optional): The maximum length of the generated text. Defaults to 200.

        device (str, optional): The device on which the model is to be run. Defaults to "cpu".

        low_memory_setting (bool, optional): Whether to use low memory setting. Defaults to True.

    Returns:
        str: The text generated by the model
    """

    model = T5ForConditionalGeneration.from_pretrained("grammarly/coedit-large")
    tokenizer = AutoTokenizer.from_pretrained("grammarly/coedit-large")

    if task == "grammar":
        prompt = "Fix the grammar: {text}"
    elif task == "coherent":
        prompt = "Make this text coherent: {text}"
    elif task == "simpler":
        prompt = "Rewrite to make this easier to understand: {text}"
    elif task == "paraphrase":
        prompt = "Paraphrase this: {text}"
    elif task == "formal":
        prompt = "Write this more formally: {text}"
    elif task == "neutral":
        prompt = "Write in a more neutral way: {text}"
    else:
        prompt = "Fix the grammar: {text}"

    input_text_list = re.split(
        r"\.\s", input_text
    )  # split the input text into sentences

    # Prepend the transformational prompt to each sentence in the input text
    prompts = [prompt.format(text=sentence) for sentence in input_text_list]

    input_batch = tokenizer.batch_encode_plus(
        prompts, return_tensors="pt", padding=True, truncation=True
    )

    model = model.to(device)
    model.eval()
    input_batch = input_batch.to(device)

    # Model generation settings
    setting = dict(
        max_length=max_length,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        low_memory=low_memory_setting,
    )

    if (
        decoding_strategy == "stochastic" or decoding_strategy == "s"
    ):  # do nucleus sampling
        setting["do_sample"] = True
        setting["top_p"] = 0.95
    else:  # do greedy decoding
        setting["do_sample"] = False
        setting["num_beams"] = 1

    ouptut_batch = model.generate(
        **input_batch,
        **setting,
    )

    generated_texts = tokenizer.batch_decode(ouptut_batch, skip_special_tokens=True)

    return "Generated: " + " ".join(generated_texts)


def predict_suggestions(
    input_text: str,
    num_token_suggestions: int = 5,
    token_suggestion_qty: int = 1,
    decoding_strategy: str = "stochastic",
    device="cpu",
    low_memory_setting=True,
) -> tuple:
    """
    This function generates token suggestions based on the input text provided.

    The function uses the GPT-2 model to generate token suggestions based on the input text provided.
    The model is fine-tuned on the GPT-2 architecture and is capable of generating text based on the
    input text. The function allows for different decoding strategies such as nucleus sampling and
    greedy decoding. The generated token suggestions can be used to complete the sentences or
    generate new text based on the input text.

    Args:
        input_text (str): The input text for which token suggestions are to be generated

        num_token_suggestions (int, optional): The number of token suggestions to be generated.
        Defaults to 5.

        token_suggestion_qty (int, optional): The number of tokens to be suggested in each
        suggestion. Defaults to 1.

        decoding_strategy (str, optional): The decoding strategy to be used by the model.
        Options are: "stochastic" or "s" for nucleus sampling, "greedy" or "g" for greedy decoding.
        Defaults to "stochastic".

        device (str, optional): The device on which the model is to be run. Defaults to "cpu".

        low_memory_setting (bool, optional): Whether to use low memory setting. Defaults to True.

    Returns:
        str: A tuple of token suggestions
    """

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    model = model.to(device)
    model.eval()

    prompt = f"Complete the sentences keeping the context intact: {input_text}"

    tokenized = tokenizer.encode(prompt, return_tensors="pt")
    tokenized = tokenized.to(device)

    # Model generation settings
    setting = dict(
        max_length=len(tokenized[0]) + token_suggestion_qty,
        num_return_sequences=num_token_suggestions,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        low_memory=low_memory_setting,
    )

    if (
        decoding_strategy == "stochastic" or decoding_strategy == "s"
    ):  # do nucleus sampling
        setting["do_sample"] = True
        setting["top_p"] = 0.95
    else:  # do greedy decoding
        setting["do_sample"] = False
        setting["num_beams"] = (
            num_token_suggestions  # return all beams whose qty is equal to num_token_suggestions
        )

    outputs = model.generate(
        tokenized,
        **setting,
    )

    final_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    # final_tokens = tuple(
    #     {re.split(input_text + r"\s?", text)[-1] for text in final_outputs} # ignores any whitespace or newline characters
    # )
    # final_tokens = list({text.replace(prompt, "").strip() for text in final_outputs}) # ignores any whitespace or newline characters
    final_tokens = list({text.replace(prompt, "") for text in final_outputs})
    # final_tokens = list(
    #     "<newline>" if token == "\n" else token for token in final_tokens
    # ) # replace newline characters with "<newline>"
    try:
        final_tokens.remove("")  # remove empty strings
    except ValueError:
        pass

    return tuple(final_tokens)