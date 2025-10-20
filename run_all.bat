@echo off
echo ===============================================
echo     🚀 Restaurant Discovery Chatbot Setup
echo ===============================================


REM --- Step 5: Clean and preprocess data ---
echo 🧹 Cleaning and preprocessing data...
python backend\data_processing.py

REM --- Step 6: Generate embeddings ---
echo 🧠 Generating embeddings...
python backend\semantic_search.py

REM --- Step 7: Index data in Typesense ---
echo 🔍 Setting up Typesense collection...
python backend\typesense_setup.py

REM --- Step 8: Launch the chatbot server ---
echo 💬 Starting Flask WhatsApp Chatbot...
python backend\app.py

pause
