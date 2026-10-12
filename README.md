# AI Chatbot — IT Helpdesk Assistant

## Overview
A functional AI chatbot built in Python 3.x using NLTK for IT Helpdesk support. It features 10 distinct intents, natural language input preprocessing, context-aware multi-turn conversations, fallback handling for unrecognized queries, a Command Line Interface (CLI), and automatic session logging.

## Tech Stack
- **Language:** Python 3.x
- **NLP Library:** NLTK (Tokenization & WordNet Lemmatization)
- **Framework/Interface:** Command Line Interface (CLI)
- **Dataset Format:** JSON (`intents.json`)

## Setup and Run Instructions
1. Place `chatbot.py` and `intents.json` in the same folder.
2. Install the required NLTK library:
   ```bash
   pip install nltk