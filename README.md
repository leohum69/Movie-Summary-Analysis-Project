# Movie Summary Analysis Project

## Overview

This project provides a Movie Summary Analyzer with a graphical user interface (GUI) built using Python and Tkinter. It predicts movie genres from a given summary, analyzes sentiment, and can convert summaries to audio in multiple languages.

## Features

- **Genre Prediction:** Uses machine learning to predict genres from movie summaries.
- **Sentiment Analysis:** Analyzes the sentiment of the summary.
- **Text Preprocessing:** Tokenization, lemmatization, and stopword removal.
- **Audio Conversion:** Converts summaries to speech in various languages using Google Text-to-Speech.
- **GUI:** User-friendly interface for input and results.

## Files

- `App.py`: Main Python application with the GUI and all core logic.
- `Project_22i-1749_22i-0518.ipynb`: Jupyter notebook with data processing, model training, and experiments.
- `Project_22i-1749_22i-0518.pdf`: Project report/documentation.
- `README.md`: This file.

## Requirements

- Python 3.x
- Libraries: `tkinter`, `numpy`, `pandas`, `scikit-learn`, `nltk`, `scipy`, `gtts`, `deep_translator`, `tqdm`
- Pre-trained model files in a `models/` directory (see `App.py` for expected filenames).

## Usage

1. Install required libraries:
   ```
   pip install -r requirements.txt
   ```
2. Ensure the `models/` directory contains the necessary `.pkl` files.
3. Run the application:
   ```
   python App.py
   ```

## Authors

- 22i-1749
- 22i-0518
