# Medical NLP Spellchecker
A medical-aware spell checking and context correction system built using Natural Language Processing (NLP), statistical language models, and efficient search data structures. The application detects spelling errors, contextual mistakes, and grammatical variations in medical text and provides ranked correction suggestions through an interactive graphical interface.
The system integrates SpaCy, NLTK, BK-Trees, and probabilistic language models to analyze text while taking into account medical vocabulary and linguistic context.

# Features
- Detects non-word spelling errors (misspelled words not in vocabulary)
- Detects contextual errors where a valid word is used incorrectly
- Handles tense and plural variations using morphological analysis
- Uses a medical glossary extracted from a PDF source

# System Architecture
- Tokenization & POS Tagging
- Medical Vocabulary Validation
- Error Detection
- Candidate Ranking

# Technologies Used
- Python
- SpaCy
- NLTK
- PyMuPDF
- Tkinter
- Levenshtein Edit Distance
- BK-Tree Data Structure
