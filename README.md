# Shakespeare Chatbot Project

**Course: CSCI933 - Machine Learning Algorithms and Applications**

---

## Table of Contents

- [About This Project](#about-this-project)
- [What It Does](#what-it-does)
- [How to Run It](#how-to-run-it)
  - [Prerequisites](#prerequisites)
  - [Installation Steps](#installation-steps)
- [How to Use It](#how-to-use-it)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [How to Develop the Chatbot](#how-to-develop-the-chatbot)
- [Contact](#contact)

---

## About This Project

This project is a domain-specific chatbot designed to engage users in conversations about Shakespearean literature. Developed as part of the CSCI933 course assignment, it aims to demonstrate foundational concepts in natural language processing (NLP) and interactive application development. The chatbot leverages pre-trained language models and retrieval-augmented generation to provide accurate responses about Shakespeare’s plays, including factual answers, thematic analyses, quote generation, and scene summaries. The project showcases skills in NLP, Python programming, and web-based application deployment using Streamlit.

---

## What It Does

The Shakespeare Chatbot offers the following functionalities:

- **Multi-Turn Dialogue**: Engages in contextual conversations about Shakespeare’s works, maintaining coherence across multiple user inputs.
- **Factual and Thematic Queries**: Answers questions about characters, plots, and themes (e.g., "Who is Hamlet’s mother?" or "What is the theme of ambition in Macbeth?").
- **Quote Generation**: Provides relevant quotes from Shakespeare’s plays based on user prompts (e.g., "Give me a quote about love from Romeo and Juliet").
- **Scene Summaries**: Generates concise prose summaries of specific scenes when provided with the play and scene details (e.g., "Summarize Act 3, Scene 1 of Hamlet").

---

## How to Run It

This section explains how to set up and run the chatbot on your local machine.

### Prerequisites

Ensure the following software is installed:

- **Python 3.12**: The programming language used for the chatbot.
- **pip**: Python’s package manager for installing dependencies.
- **Git**: For cloning the project repository.
- **Virtualenv** (optional but recommended): To create an isolated Python environment.

### Installation Steps

1. **Download the project**:

   - Download and extract the project zip file from the provided source
   - Or download repository

   ```bash
   git clone [https://github.com/id-nynt/CSCI933_shakespeare_chatbot.git]
   cd CSCI933_shakespeare_chatbot

   ```

2. **Set Up a Virtual Environment** (optional):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Install the required Python libraries listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Chatbot**:

   - Launch the Streamlit web interface:
     ```bash
     streamlit run app.py
     ```
   - Alternatively, use the command-line interface (CLI):
     ```bash
     python cli.py
     ```

5. **Access the Chatbot**:
   - For Streamlit, open the provided local URL (e.g., `http://localhost:8501`) in a web browser.
   - For the CLI, interact directly in the terminal.

---

## How to Use It

- **Streamlit Interface**:

  1. Run `streamlit run app.py` to start the web server.
  2. Open the URL displayed in the terminal (e.g., `http://localhost:8501`) in a web browser.
  3. Enter your question or prompt in the text input field (e.g., "Summarize Act 1, Scene 1 of King Lear").
  4. Click **Submit** to receive the chatbot’s response.
  5. Continue the conversation by entering follow-up questions.

- **CLI Interface**:
  1. Run `python cli.py` to start the terminal-based chatbot.
  2. Type your question or prompt and press **Enter**.
  3. The chatbot responds in the terminal, and you can continue the dialogue.

---

## Technologies Used

The project utilizes the following tools and libraries:

- **Python 3.12**: Core programming language for implementing the chatbot’s logic.
- **HuggingFace Transformers**: Provides pre-trained language models (e.g., BERT or GPT-based models) for natural language understanding and generation.
- **LangChain**: Facilitates the integration of language models with external data sources and conversation memory for coherent multi-turn dialogues.
- **Streamlit**: Enables the creation of an interactive web-based user interface.
- **Other Libraries** (listed in `requirements.txt`): Includes dependencies like `numpy` for data processing and `pandas` for handling structured data, if applicable.

These technologies were chosen to balance performance, ease of development, and accessibility for a domain-specific NLP application.

---

## Project Structure

Below is the organization of the project’s files and directories:

```
shakespeare_chatbot/
├── app.py                  # Main Streamlit application script
├── cli.py                  # Command-line interface script
├── requirements.txt        # List of Python dependencies
├── data/                   # Directory for dataset
├── retrieval/              # RAG system
├── models/                 # Directory for fine-tuned model
├── utils/                  # Utility scripts for deployment
├── scripts/                # Scripts for text processing and development
└── README.md               # Project documentation (this file)
```

---

## Future Improvements

To enhance the chatbot’s functionality and robustness, the following improvements are proposed:

- **Improved Accuracy**: Fine-tune the language model on a larger Shakespearean corpus to enhance response precision.
- **Complex Sentence Understanding**: Implement advanced NLP techniques, such as dependency parsing, to handle complex user queries.
- **Contextual Memory**: Enhance LangChain’s memory capabilities to maintain longer conversation contexts.
- **Multimodal Features**: Add support for visualizing character relationships or scene timelines using libraries like Plotly.
- **Error Handling**: Improve robustness by handling edge cases, such as ambiguous or off-topic user inputs.

---
