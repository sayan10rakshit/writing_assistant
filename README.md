# Writing Assistant

## Enhance Your Writing with the AI-Powered Writing Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://writingassistant.streamlit.app/)

### Introduction

This web application empowers you to elevate the quality of your written work by providing intelligent suggestions and corrections. Leverage its capabilities to refine your writing by identifying opportunities for improved vocabulary, correcting grammatical errors, and more. The Writing Assistant is an invaluable tool for anyone seeking to strengthen their writing skills.

## Features

- **Vocabulary Enrichment**: Receive suggestions for superior word choices to enhance the impact of your text.
- **Grammar Perfection**: Eliminate grammatical inconsistencies and ensure a polished writing style.
- **Sentence Refinement**: Optimize sentence construction for improved clarity and conciseness.
- **Punctuation Precision**: Eradicate punctuation errors for a professional presentation.
- **Spelling Accuracy**: Ensure flawless spelling throughout your documents.
- **Enhanced Coherence**: Foster a clear and consistent flow of ideas within your writing.
- **Tonal Adjustment**: Tailor the voice of your writing to achieve a formal or neutral tone as required.
- **Token Suggestions**: Set the desired number of token suggestions to receive for each input.
- **Words per suggestion**: Set the desired number of words per suggestion to receive for each input.

## Technical Underpinnings

- **Cutting-Edge 🤗Transformers**: Leverage the power of state-of-the-art Transformers for intelligent text processing.
- **🤖 PyTorch Deep Learning Framework**: Harness the capabilities of the PyTorch framework for efficient deep learning implementations.
- **User-Friendly 🎈Streamlit Interface**: Enjoy a seamless and intuitive user experience facilitated by the Streamlit library.

## Model Architecture

- **GPT-2**: The GPT-2 decoder only model, a pre-trained powerhouse for text generation, to offer insightful suggestions.
- **T5**: Utilizes a T5 encoder-decoder model, pre-trained and further finetuned on the Coedit dataset, for exceptional text correction capabilities.

## Getting Started

1. **Install pipenv**:

```bash
pip install pipenv
```

2. **Set Up Dependencies**

```bash
pipenv shell
pipenv install
```

3. **Launch the App**

```bash
streamlit run app.py
```
