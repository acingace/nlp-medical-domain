import json
import pickle
import re
import os
from collections import Counter
import nltk
from nltk.tokenize import PunktSentenceTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# --- CONFIGURATION ---
INPUT_JSONL = "/Users/gracegomes/Desktop/SpellCheck/Medical_Domain_Corpus.jsonl"
CONVO_FILE = "/Users/gracegomes/Desktop/SpellCheck/dictionary.txt" 
OUTPUT_MODEL = "medical_model.pkl"
WORD_LIMIT = 2000000 

class MedicalModelBuilder:
    def __init__(self):
        self.unigram = Counter()
        self.bigram = Counter()
        self.trigram = Counter()
        self.tokenizer = PunktSentenceTokenizer()
        self.lemmatizer = WordNetLemmatizer()
        self.dictionary_set = set()
        self.total_medical_words = 0

    def load_base_dictionaries(self):
        print("Loading Essential English Stopwords...")
        try:
            self.dictionary_set.update(set(w.lower() for w in stopwords.words('english')))
        except:
            nltk.download('stopwords')
            self.dictionary_set.update(set(w.lower() for w in stopwords.words('english')))

    def update_ngrams(self, tokens):
        toks = ['<s>', '<s>'] + tokens + ['</s>']
        for i in range(len(toks)):
            self.unigram[toks[i]] += 1
            if i + 1 < len(toks):
                self.bigram[(toks[i], toks[i+1])] += 1
            if i + 2 < len(toks):
                self.trigram[(toks[i], toks[i+1], toks[i+2])] += 1

    def process_text_file(self, path):
        print(f"Learning from Clinical Conversations: {path}...")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found.")
            return
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read().lower()
            sents = self.tokenizer.tokenize(content)
            for sent in sents:
                tokens = re.findall(r"[\w']+|[^\s\w]+", sent)
                self.update_ngrams(tokens)
                for t in tokens:
                    if t.isalpha():
                        self.dictionary_set.add(t)
                        self.dictionary_set.add(self.lemmatizer.lemmatize(t))

    def train(self):
        self.load_base_dictionaries()
        self.process_text_file(CONVO_FILE)

        print(f"Checking {INPUT_JSONL}...")
        if not os.path.exists(INPUT_JSONL):
            print(f"ERROR: JSONL file not found at {INPUT_JSONL}")
            return

        with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if self.total_medical_words >= WORD_LIMIT:
                    break
                
                try:
                    data = json.loads(line)
                    
                    # --- DEBUG: Print keys for the first line only ---
                    if line_num == 0:
                        print(f"Keys found in your JSONL: {list(data.keys())}")
                    
                    # Extract ALL string values from the JSON object
                    content = " ".join([str(v) for v in data.values() if isinstance(v, str)]).lower()
                    
                    if not content.strip():
                        continue

                    sents = self.tokenizer.tokenize(content)
                    for sent in sents:
                        tokens = re.findall(r"[\w']+|[^\s\w]+", sent)
                        
                        if self.total_medical_words + len(tokens) > WORD_LIMIT:
                            tokens = tokens[:WORD_LIMIT - self.total_medical_words]
                        
                        if not tokens: break

                        self.update_ngrams(tokens)
                        self.total_medical_words += len(tokens)
                        
                        for t in tokens:
                            if t.isalpha() and self.unigram[t] > 1:
                                self.dictionary_set.add(t)
                                self.dictionary_set.add(self.lemmatizer.lemmatize(t))
                                
                except Exception as e:
                    if line_num < 5: print(f"Error on line {line_num}: {e}")
                    continue
        
        self.save_model()

    def save_model(self):
        model_data = {'uni': self.unigram, 'bi': self.bigram, 'tri': self.trigram, 'vocab': self.dictionary_set}
        with open(OUTPUT_MODEL, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"\n--- Training Complete ---")
        print(f"Medical Words Processed: {self.total_medical_words}")
        print(f"Final Vocab Size: {len(self.dictionary_set)}")

if __name__ == "__main__":
    builder = MedicalModelBuilder()
    builder.train()