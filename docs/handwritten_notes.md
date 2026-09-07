# NEXORA — Handwritten Theory Notes

**Machine Translation using Sequence-to-Sequence Networks with Attention-Based Alignment**
*(Concise notes designed for handwritten A4 pages.)*

---

## 1. Problem Statement

Machine translation = automatic conversion of text from one language to
another. Classic MT needs huge hand-built pipelines and fails on long sentences.
We need a model that: (i) reads a variable-length source sentence, (ii)
generates a variable-length target sentence, (iii) aligns source words to
target words so long sentences translate well, and (iv) works for many
languages. NEXORA solves (i)–(iv) with a Seq2Seq network + attention, plus
classical ML baselines (language ID) for comparison.

## 2. Dataset

**Samanantar** (AI4Bharat, IIT Madras) — largest public parallel corpus for 11
Indic languages. Format per language config: `idx`, `src` (English), `tgt`
(target). We use Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu.
URL: huggingface.co/datasets/ai4bharat/samanantar.

## 3. Preprocessing Techniques

- Missing values → drop + count.
- Duplicate rows → remove + count.
- Unicode NFC normalization (merges equivalent characters).
- Remove control chars; collapse whitespace.
- **Script-consistency filter** — row must contain its claimed script.
- No aggressive lowercasing/character removal for Indic scripts (preserves
  matras, nukta, Tamil/Telugu vowel signs).
- Tokenization (Unicode-aware, punctuation kept).
- Build vocabulary with `<sos> <eos> <pad> <unk>`.
- Pad + truncate to max_len (loss ignores padding).

## 4. Feature Engineering

Classical models need numbers → convert text to vectors:

### Bag of Words (BoW)
Count occurrences of each word per document.
`X[i][j] = count of word j in doc i`.
Simple, sparse, interpretable. Ignores word order & semantics.

### CountVectorizer
Scikit-learn tool that produces the BoW matrix. Options: ngram_range,
max_features (vocab cap), custom analyzer (algorithm that extracts tokens).

### TF-IDF
Term Frequency × Inverse Document Frequency:

```
tf-idf(t,d) = tf(t,d) × log( N / df(t) )
```

- `tf(t,d)` = how often term t appears in doc d.
- `df(t)` = how many docs contain t.
Down-weights common words, boosts rare discriminative words → ideal for
language ID. We also use **character n-grams (n=2–4)** to catch script
fingerprints.

## 5. Classical ML Algorithms (Language Identification)

Task: given a sentence, predict language (Hi/Mr/Bn/Gu/Ta/Te).

### Naïve Bayes
Bayes' rule with independence assumption:

```
P(C|X) ∝ P(C) × ∏ P(x_i | C)
```

Generative, fast, works well on text. Smoothing (alpha) avoids zero
probabilities.

### Logistic Regression
Discriminative linear classifier; multiclass uses softmax:

```
P(C=c | x) = e^{w_c·x} / Σ_c' e^{w_c'·x}
```

L2 regularization prevents overfitting.

### SVM (Support Vector Machine)
Max-margin classifier: finds hyperplane with largest margin between classes.
LinearSVC = one-vs-rest linear SVM for multiclass. Robust for high-dimensional
sparse text features.

**Metrics:** accuracy, precision, recall, F1, confusion matrix, ROC AUC.

```
Precision = TP/(TP+FP)   Recall = TP/(TP+FN)
F1 = 2PR/(P+R)
```

## 6. Sequence-to-Sequence Networks

Map source `x1..xm` → target `y1..yn`:

```
P(y|x) = Π_t P(y_t | y_<t, x)
```

Two parts: **Encoder** (reads source) and **Decoder** (writes target).

## 7. Encoder

- Input: source sentence.
- Each word → embedding vector (learned).
- Process with bi-directional **LSTM** → hidden states h1..hm.
- Bidirectional: forward (left→right) + backward (right→left), averaged.
- Outputs the *sequence* of hidden states (not one vector) — attention uses all
  of them.

## 8. Decoder

- Autoregressive: generates target word by word.
- **GRU** (simpler LSTM variant, fewer parameters).
- At step t: input = previous target embedding + **context vector** c_t:

```
s_t = GRU( [y_{t-1} ; c_t], s_{t-1} )
logit_t = W_o [s_t ; c_t] + b_o
y_t = argmax softmax(logit_t)
```

- Stops at `<eos>` or max length.

## 9. LSTM / GRU

RNN variants that fix the vanishing/exploding gradient problem with gates.

**LSTM:** input, forget, output gates + cell state:

```
f_t = σ(W_f x_t + U_f h_{t-1} + b_f)   (forget)
i_t = σ(W_i x_t + U_i h_{t-1} + b_i)   (input)
o_t = σ(W_o x_t + U_o h_{t-1} + b_o)   (output)
c̃_t = tanh(W_c x_t + U_c h_{t-1} + b_c)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
h_t = o_t ⊙ tanh(c_t)
```

**GRU:** merges input+forget into a single update gate, drops the cell state —
faster, comparable quality.

## 10. Attention Mechanism (Bahdanau / Additive)

Problem: fixed context vector can't carry a long sentence. Idea: at each
decoding step, recompute the context from ALL source hidden states.

1. **Score** decoder state vs each encoder state:

```
e_tj = vᵀ tanh( W_a h_j + U_a s_{t-1} )
```

2. **Weights** = softmax of scores:

```
α_tj = exp(e_tj) / Σ_j' exp(e_tj')
```

3. **Context vector** = weighted sum:

```
c_t = Σ_j α_tj h_j
```

Attention weights = which source words the decoder attends to.

## 11. Attention-Based Alignment

Alignment = correspondence between source and target tokens.
Attention gives a **soft alignment**: each target token attends to all source
tokens with weights α. Read as a matrix (rows = target, cols = source):
monotone-ish diagonal structure in well-translated sentences; shifts show
reordering.

## 12. Evaluation Metrics

- **BLEU:** n-gram overlap between prediction and reference, with brevity
  penalty. Range 0–1 (often ×100). Corpus BLEU = geometric mean of n-gram
  precisions (1-4).
- **chrF:** character-level F-score — robust to morphology (noted as future
  work here).
- **Loss curves:** training + validation cross-entropy over epochs.
- **Classical:** accuracy, precision, recall, F1, confusion matrix, ROC AUC.

## 13. NEXORA System

- **Production engine:** facebook/m2m100_418M — pretrained, 100 languages
  (NOT trained by us; clearly labelled).
- **Experimental model:** BiLSTM encoder + GRU decoder + Bahdanau attention,
  trained on English→Hindi sample (small vocab, few epochs, CPU) — this model
  is ours and drives the **attention heatmaps**.
- **Dashboard:** Streamlit — Home, Live Translation, Attention Explorer,
  Dataset & EDA, Classical ML, Seq2Seq Neural Model, About/Conclusion.

## 14. Key Takeaways

1. Seq2Seq + attention = read everything, align softly, generate word by word.
2. Attention is learnable and visualizable → real heatmaps.
3. Classical BoW/TF-IDF features are powerful for text classification
   (language ID ~ near-perfect).
4. Separate production (pretrained) from experimental (trained) honestly.
5. Report only measured numbers: accuracy, F1, BLEU, loss.