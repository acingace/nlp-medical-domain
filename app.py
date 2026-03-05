import os
import math
import re
import tkinter as tk
from tkinter import ttk, messagebox
from collections import Counter
import nltk
import spacy
import fitz  # PyMuPDF
from nltk.data import path as nltk_data_path

# --- Setup Spacy and NLTK ---
# Using 'en_core_web_sm' for speed; 'en_core_web_md' is better for accuracy if downloaded.
try:
    nlp = spacy.load("en_core_sci_md")
except OSError:
    print("Medical model not found. Use the link above to install it.")
    nlp = spacy.load("en_core_web_sm")

user_nltk_dir = os.path.expanduser('~/nltk_data')
if user_nltk_dir not in nltk_data_path:
    nltk_data_path.append(user_nltk_dir)

# ---------------------------
# AMA PDF GLOSSARY PARSER
# ---------------------------
GLOSSARY_PATH = "/Users/gracegomes/Desktop/SpellCheck/Glossary.pdf"

def load_ama_glossary(path):
    anchors = set()
    if os.path.exists(path):
        try:
            doc = fitz.open(path)
            for page in doc:
                text = page.get_text().lower()
                words = re.findall(r"\b[a-z]{3,}\b", text)
                anchors.update(words)
        except Exception as e:
            print(f"Error reading PDF: {e}")
    return anchors

MEDICAL_ANCHORS = load_ama_glossary(GLOSSARY_PATH)
STOPWORDS = nlp.Defaults.stop_words

# ---------------------------
# IMPROVED MORPHOLOGICAL CHECK
# ---------------------------
def is_known_word(word_str):
    doc = nlp(word_str.lower())
    if not doc: return False
    
    token = doc[0]
    # The medical lemma is much more accurate here
    # Example: 'ataxic' will lemma to 'ataxia'
    lemma = token.lemma_
    
    # 1. Direct Check
    if word_str.lower() in WORDSET or word_str.lower() in MEDICAL_ANCHORS:
        return True
    
    # 2. Medical Root Check (Specialist Lexicon Logic)
    if lemma in MEDICAL_ANCHORS or lemma in WORDSET:
        return True
        
    # 3. Suffix Check for medical Greek/Latin plurals not in standard WordNet
    # This mimics the Specialist Lexicon's rules
    medical_suffixes = ('es', 'ia', 'a', 'i')
    if word_str.lower().endswith(medical_suffixes):
        # Logic for things like: diagnosis -> diagnoses, bacterium -> bacteria
        return True 

    return False

# ---------------------------
# MODEL LOADING
# ---------------------------
CORPUS_PATH = "dictionary.txt" 

def build_models():
    if os.path.exists(CORPUS_PATH):
        with open(CORPUS_PATH, "r", encoding="utf8") as f:
            text = f.read().lower()
    else:
        from nltk.corpus import brown
        text = "\n".join([" ".join(s) for s in nltk.corpus.brown.sents()]).lower()

    tokens = re.findall(r"[\w']+", text)
    unigram = Counter(tokens)
    bigram = Counter(zip(tokens, tokens[1:]))
    return unigram, bigram, set(unigram.keys())

UNIGRAM, BIGRAM, WORDSET = build_models()
WORD_LIST = sorted(list(WORDSET))

# ---------------------------
# PROBABILITY & POS RANKING
# ---------------------------
def get_context_prob(w1, w2):
    count_pair = BIGRAM.get((w1, w2), 0)
    count_w1 = UNIGRAM.get(w1, 0)
    if count_pair > 0:
        return (count_pair + 0.1) / (count_w1 + len(UNIGRAM))
    return (UNIGRAM.get(w2, 0) + 0.001) / (sum(UNIGRAM.values()) * 10)

def candidate_ranking_score(dist, word, prev, nxt, target_pos):
    p_l = get_context_prob(prev, word)
    p_r = get_context_prob(word, nxt)
    ctx_cost = -math.log(((p_l + p_r) / 2.0) + 1e-6)
    
    # POS Tagging for the candidate
    cand_doc = nlp(word)
    cand_pos = cand_doc[0].pos_ if len(cand_doc) > 0 else ""
    
    # Penalty if the candidate isn't the same part of speech (e.g., verb vs noun)
    pos_penalty = 1.0 if cand_pos == target_pos else 2.5
    
    boost = 0.7 if word in MEDICAL_ANCHORS else 1.0
    return ((dist * 1.5) + (ctx_cost * 15.0)) * boost * pos_penalty

# ---------------------------
# BK-TREE & EDIT DISTANCE
# ---------------------------
def edit_distance(a, b):
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1): dp[i][0] = i
    for j in range(lb+1): dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1]==b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
    return dp[la][lb]

class BKNode:
    def __init__(self, word):
        self.word, self.children = word, {}

class BKTree:
    def __init__(self, distfn):
        self.distfn, self.root = distfn, None
    def add(self, word):
        if not self.root: self.root = BKNode(word); return
        node = self.root
        while True:
            d = self.distfn(word, node.word)
            if d in node.children: node = node.children[d]
            else: node.children[d] = BKNode(word); break
    def build(self, words):
        for w in words: self.add(w)
    def find(self, word, max_dist):
        if not self.root: return []
        results, nodes = [], [self.root]
        while nodes:
            node = nodes.pop()
            d = self.distfn(word, node.word)
            if d <= max_dist: results.append((node.word, d))
            low, high = d - max_dist, d + max_dist
            for child_d, child in node.children.items():
                if low <= child_d <= high: nodes.append(child)
        return results

BK = BKTree(distfn=edit_distance)
BK.build(WORD_LIST[:80000])

# ---------------------------
# GUI APPLICATION
# ---------------------------
class SpellApp:
    def __init__(self, root):
        self.approved_words = set()
        self.root = root
        root.title("Medical NLP Spellchecker (Tense & Plural Aware)")
        root.geometry("1200x800")

        self.paned = ttk.PanedWindow(root, orient='horizontal')
        self.paned.pack(fill='both', expand=True)

        self.left_frame = ttk.Frame(self.paned, padding=5)
        self.paned.add(self.left_frame, weight=4)

        self.text = tk.Text(self.left_frame, height=12, font=('Helvetica', 12), wrap='word')
        self.text.pack(fill='x', padx=10, pady=10)
        
        btn_frame = ttk.Frame(self.left_frame)
        btn_frame.pack(fill='x', padx=10)
        ttk.Button(btn_frame, text="Analyze Content", command=self.check_spelling).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Clear", command=self.clear_all).pack(side='left')

        self.results = tk.Text(self.left_frame, bg='#1a1a1a', fg='#00ff41', font=('Courier New', 10))
        self.results.pack(fill='both', expand=True, padx=10, pady=10)

        self.right_frame = tk.Frame(self.paned, bg='#1a1a1a', padx=10, pady=10)
        self.paned.add(self.right_frame, weight=1)

        tk.Label(self.right_frame, text="Medical Vocabulary", bg='#1a1a1a', fg='#00ff41').pack()
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(self.right_frame, textvariable=self.search_var)
        self.search_entry.pack(fill='x', pady=5)
        self.search_entry.bind('<KeyRelease>', self.filter_vocab)

        self.vocab_listbox = tk.Listbox(self.right_frame, bg='#1a1a1a', fg='#00ff41')
        self.vocab_listbox.pack(fill='both', expand=True)
        
        self.text.tag_configure("miss", background="#ffcccc", underline=True, foreground="red")
        self.text.tag_bind("miss", "<Button-1>", self.on_word_click)
        self.fill_vocab_list(WORD_LIST[:100])

    def fill_vocab_list(self, words):
        self.vocab_listbox.delete(0, 'end')
        for w in words:
            self.vocab_listbox.insert('end', w)

    def filter_vocab(self, event):
        query = self.search_var.get().lower()
        filtered = [w for w in WORD_LIST if w.startswith(query)][:100]
        self.fill_vocab_list(filtered)

    def clear_all(self):
        self.text.delete('1.0', 'end')
        self.results.configure(state='normal')
        self.results.delete('1.0', 'end')
        self.results.configure(state='disabled')

    def check_spelling(self):
        content = self.text.get("1.0", "end-1c")
        doc = nlp(content)
        
        self.text.tag_remove("miss", "1.0", "end")
        self.results.configure(state='normal')
        self.results.delete('1.0', 'end')

        tokens = [t for t in doc if not t.is_space]
        
        for i, token in enumerate(tokens):
            tok_text = token.text
            tok_l = tok_text.lower()
            
            if not token.is_alpha or tok_l in STOPWORDS: continue
            if tok_l in self.approved_words: continue

            start_idx = f"1.0 + {token.idx} chars"
            end_idx = f"1.0 + {token.idx + len(tok_text)} chars"
            
            # --- 1. NON-WORD ERROR CHECK ---
            if not is_known_word(tok_l):
                self.text.tag_add("miss", start_idx, end_idx)
                self.results.insert('end', f"[SPELLING] '{tok_text}': Word not recognized.\n")
                continue

            # --- 2. CONTEXT / GRAMMAR CHECK ---
            # We look at the probability of this word appearing between its neighbors
            prev = tokens[i-1].text.lower() if i > 0 else '<s>'
            nxt = tokens[i+1].text.lower() if i < len(tokens)-1 else '</s>'
            
            current_prob = (get_context_prob(prev, tok_l) + get_context_prob(tok_l, nxt)) / 2.0
            
            # Look for better candidates that fit the grammar (POS) better
            candidates = BK.find(tok_l, 1) # Edit distance 1 to find close swaps (e.g., patient/presents)
            for cand_w, dist in candidates:
                if cand_w == tok_l: continue
                
                # Check if candidate fits the POS context better
                cand_prob = (get_context_prob(prev, cand_w) + get_context_prob(cand_w, nxt)) / 2.0
                
                # If a nearby word is 100x more likely in this context, flag it
                if cand_prob > (current_prob * 100):
                    self.text.tag_add("miss", start_idx, end_idx)
                    # Distinguish between Tense and Context
                    cand_doc = nlp(cand_w)
                    if cand_doc[0].lemma_ == token.lemma_:
                        self.results.insert('end', f"[GRAMMAR] '{tok_text}': Suggest '{cand_w}' (Tense/Plural issue).\n")
                    else:
                        self.results.insert('end', f"[CONTEXT] '{tok_text}': Suggest '{cand_w}' based on surroundings.\n")
                    break

        self.results.insert('end', "\n--- Analysis Finished ---")
        self.results.configure(state='disabled')

    def on_word_click(self, event):
        idx = self.text.index(f"@{event.x},{event.y}")
        content = self.text.get("1.0", "end-1c")
        doc = nlp(content)
        
        # Get character index accurately
        char_idx = self.text.count("1.0", idx, "chars")[0]
        
        target_token = None
        for t in doc:
            if t.idx <= char_idx < (t.idx + len(t.text)):
                target_token = t
                break
        
        if not target_token: return
        
        word = target_token.text
        pos = target_token.pos_
        prev = doc[target_token.i - 1].text.lower() if target_token.i > 0 else '<s>'
        nxt = doc[target_token.i + 1].text.lower() if target_token.i < len(doc)-1 else '</s>'
        
        # Create Popup
        pop = tk.Toplevel(self.root)
        pop.title(f"Correction: {word}")
        pop.geometry("450x500")
        pop.configure(bg='#1a1a1a')

        # --- Suggestions Listbox ---
        tk.Label(pop, text=f"Suggestions for '{word}'", bg='#1a1a1a', fg='#00ff41').pack(pady=5)
        lb = tk.Listbox(pop, font=('Courier New', 11), bg='#2e2e2e', fg='#ffffff', selectbackground='#00ff41', selectforeground='#000000')
        lb.pack(padx=10, pady=10, fill='both', expand=True)

        # Generate and Rank Candidates
        candidates = BK.find(word.lower(), 2)
        scored = []
        for w, dist in candidates:
            score = candidate_ranking_score(dist, w, prev, nxt, pos)
            scored.append((w, score))
        
        scored = sorted(scored, key=lambda x: x[1])

        for w, sc in scored[:15]:
            lb.insert('end', f"{w:<15} [Score: {sc:.2f}]")

        # --- Action Functions ---
        def apply_selection():
            if not lb.curselection():
                messagebox.showwarning("Selection", "Please select a suggestion from the list.")
                return
            raw_val = lb.get(lb.curselection()[0])
            choice = raw_val.split()[0] # Extract word from "word [Score: 0.00]"
            
            # Correct the text in the main editor
            self.text.delete(f"1.0 + {target_token.idx} chars", f"1.0 + {target_token.idx + len(word)} chars")
            self.text.insert(f"1.0 + {target_token.idx} chars", choice)
            pop.destroy()
            self.check_spelling()

        def keep_original():
            self.approved_words.add(word.lower())
            pop.destroy()
            self.check_spelling()

        # --- Control Buttons ---
        btn_frame = ttk.Frame(pop)
        btn_frame.pack(side='bottom', pady=20)

        ttk.Button(btn_frame, text="Apply Change", command=apply_selection).pack(side='left', padx=10)
        ttk.Button(btn_frame, text="Maintain", command=keep_original).pack(side='left', padx=10)
        candidates = BK.find(word.lower(), 2)
        scored = []
        for w, dist in candidates:
            score = candidate_ranking_score(dist, w, prev, nxt, pos)
            scored.append((w, score))
        
        scored = sorted(scored, key=lambda x: x[1])

        for w, sc in scored[:15]:
            lb.insert('end', f"{w:<15} (Score: {sc:.2f})")

if __name__ == "__main__":
    root = tk.Tk(); app = SpellApp(root); root.mainloop()