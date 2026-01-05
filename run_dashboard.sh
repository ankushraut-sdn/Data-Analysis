#!/bin/bash
if [ -d ".venv" ]; then
    echo "Using virtual environment..."
    .venv/bin/python -m streamlit run app.py
else
    echo "Virtual environment not found. Please create one or install dependencies."
    exit 1
fi
