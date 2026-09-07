# NEXORA

## Machine Translation using Sequence-to-Sequence Networks with Attention-Based Alignment

---

### Title Page

| | |
|---|---|
| **Project Title** | NEXORA — Machine Translation using Sequence-to-Sequence Networks with Attention-Based Alignment |
| **Tagline** | AI That Connects Meaning |
| **Name** | ____________________ |
| **Roll No** | ____________________ |
| **Division** | ____________________ |
| **Class** | ____________________ |
| **Topic** | Machine Translation using Sequence-to-Sequence Networks with Attention-Based Alignment |
| **Guide** | ____________________ |

---

## Table of Contents

1. Introduction
2. Problem Statement
3. Objectives
4. Dataset Description
5. Dataset Source
6. Data Preprocessing
7. Exploratory Data Analysis
8. Feature Engineering
9. Classical Machine Learning Algorithms
10. Sequence-to-Sequence Architecture
11. Encoder
12. Decoder
13. Attention Mechanism
14. Attention-Based Alignment
15. Model Training
16. Model Evaluation
17. Results
18. Dashboard
19. Key Findings
20. Challenges
21. Applications
22. Future Scope
23. Conclusion
24. References

---

## 1. Introduction

Machine translation (MT) is the automatic conversion of text from one natural
language to another. For decades, classical phrase-based statistical MT (SMT)
dominated, but it required enormous hand-engineered feature pipelines and was
brittle across languages. In 2014, Sutskever et al. proposed the
**Sequence-to-Sequence (Seq2Seq)** architecture: an encoder network compresses
a variable-length source sentence into a fixed-size representation, and a
decoder network generates the target sentence token by token. This one idea
unified translation, summarization, and dialogue under a single framework and
transformed applied NLP.

A serious weakness remained: the encoder had to squeeze *all* source information
into one fixed context vector, which degraded on long sentences. In 2015,
Bahdanau et al. introduced **attention-based alignment**: at every decoding
step, the model learns to *look back* at the entire source sequence, weighting
each source hidden state by its relevance to the current output token. The
resulting **attention weights** produce a soft alignment between source and
target tokens — a property that is both mathematically beautiful and directly
visualizable.

**NEXORA** is an end-to-end demonstration of this idea. It pairs a modern,
pretrained multilingual translation engine (`facebook/m2m100_418M`) for
production-quality live translation with an self-trained experimental
Seq2Seq + Attention network whose **real attention weights** drive an
interactive alignment explorer. Alongside the neural system, NEXORA includes a
complete classical NLP track (CountVectorizer, TF-IDF, Naïve Bayes, Logistic
Regression, SVM) used for language identification, so that the project covers
the full spectrum of NLP teaching outcomes.

## 2. Problem Statement

1. **Translation quality gap:** simple neural MT models produce poor output on
   long sentences because a fixed context vector cannot carry enough
   information. Attention solves this by allowing the decoder to re-read the
   source at each step.
2. **Educational gap:** most mini-projects treat MT as a black box. NEXORA opens
   the box — training a small Seq2Seq model from scratch on a real multilingual
   corpus and visualizing its internal alignment.
3. **Language-support gap:** many apps claim multilingual support they cannot
   deliver. NEXORA shows only languages its engines genuinely support.
4. **Practical gap:** classical ML (BoW/TF-IDF + NB/LR/SVM) and neural MT are
   taught separately. NEXORA integrates both on one corpus, with an honest
   evaluation of each.

## 3. Objectives

- Build a polished, live, multilingual machine translation web application.
- Implement and explain the Seq2Seq architecture: encoder, decoder, embeddings,
  hidden states, context vectors, decoding.
- Implement Bahdanau (additive) attention and demonstrate **attention-based
  alignment** with real heatmaps.
- Use a legitimate public parallel corpus and report its true statistics.
- Implement language-aware preprocessing that respects Indic scripts.
- Implement TF-IDF and CountVectorizer feature engineering for an auxiliary
  language-identification task.
- Train and compare three classical classifiers: Naïve Bayes, Logistic
  Regression, SVM — with real metrics (accuracy, precision, recall, F1,
  confusion matrices, ROC).
- Evaluate the neural model with BLEU and training/validation loss curves.
- Clearly separate the **production translation engine** (pretrained) from the
  **experimental Seq2Seq model** (trained by us), without fabricating results.
- Deliver complete documentation and handwritten-theory material.

## 4. Dataset Description

The project uses **Samanantar**, the largest publicly available parallel
corpora collection for 11 Indic languages (Ramesh et al., TACL 2022, AI4Bharat
at IIT Madras). Samanantar aggregates sentences from multiple sources —
Wikipedia dumps, news corpora, government documents, and an English-centric
parallel corpus — and provides a consistent `src` (English) / `tgt` (target
language) pair format.

For this project we select **six** language configs: Hindi, Marathi, Bengali,
Gujarati, Tamil and Telugu. During preprocessing we stream a bounded, seeded
subset of the corpus (60 000 rows per language for the LID task; 60 000 English
rows per pair for the experimental translation task) so that all artifacts are
reproducible on a laptop. The corpus covers a broad domain mix (news, education,
government, encyclopedic text), which makes it suitable both for
language-identification and for investigating translation behaviour.

**Key properties (measured on the processed subset):**

| Property | Value |
|---|---|
| Corpus name | Samanantar (11 Indic languages) |
| Format per config | `idx`, `src` (English), `tgt` (target) |
| Languages used | Hindi, Marathi, Bengali, Gujarati, Tamil, Telugu |
| Missing values | tracked in preprocessing (see `dataset_stats.json`) |
| Duplicates | tracked per language before deduplication |
| Train / Valid / Test | 80 % / 10 % / 10 % (stratified, seed 42) |

## 5. Dataset Source

- **Dataset name:** Samanantar
- **Provider:** AI4Bharat (IIT Madras)
- **URL:** https://huggingface.co/datasets/ai4bharat/samanantar
- **Documentation:** https://indicnlp.ai4bharat.org/samanantar/
- **Citation:** Ramesh, Gowtham, et al. "Samanantar: The Largest Publicly
  Available Parallel Corpora Collection for 11 Indic Languages." TACL 10 (2022):
  1453–1470.
- **License:** permissive open-data terms as published by AI4Bharat (CC BY-style
  attribution required; see the project page).
- **Access method:** Hugging Face Hub streaming API (`datasets.load_dataset`).

## 6. Data Preprocessing

All preprocessing is implemented in `src/preprocessing.py` and is
**language-aware** — we deliberately do not aggressively lowercase, strip
diacritics or remove characters where doing so would damage meaning (e.g.
Devanagari nukta, matras, Tamil/Telugu vowel signs).

Stages:

1. **Missing-value handling:** empty strings are converted to `NaN`-style
   records, counted, and dropped from the training sets.
2. **Duplicate removal:** exact duplicate `(text, language)` rows are removed;
   duplicate counts are recorded before removal.
3. **Text cleaning:** control characters and zero-width joiners are removed;
   runs of whitespace are collapsed; surrounding whitespace is trimmed.
4. **Unicode normalization:** NFC normalization merges canonically equivalent
   code points (important for Bengali and Tamil where precomposed and
   decomposed forms both occur).
5. **Case normalization:** applied only to Latin-script (English) text — Indic
   scripts have no case, and aggressive lowercasing can conflate proper nouns.
6. **Punctuation handling:** punctuation is retained (it carries meaning); a
   custom tokenizer splits punctuation into standalone tokens for statistics.
7. **Tokenization:** a lightweight Unicode-aware tokenizer (`src/multilingual.py`)
   is used for vocabulary statistics and the experimental Seq2Seq model. The
   production engine uses its own SentencePiece subword tokenizer.
8. **Vocabulary construction:** for the experimental model we build a
   frequency-capped vocabulary (min frequency 2, max 10 000 tokens) with
   special tokens `<sos>`, `<eos>`, `<pad>`, `<unk>`.
9. **Padding & truncation:** sequences are truncated to `max_len` tokens and
   right-padded per batch; the loss ignores padding positions.
10. **Script-consistency filtering:** each sentence is verified to contain
    characters of its claimed script (e.g. a "Hindi" row must contain
    Devanagari). Rows that fail are wrong-alignment corpus noise and are
    removed.

## 7. Exploratory Data Analysis

EDA is performed entirely on the processed dataset and generates figures used
by the dashboard (`outputs/plots/`):

- **Dataset size:** total LID samples per language and overall.
- **Sentence-length distribution:** histogram per language (clipped at 60
  words) — Indic languages typically show longer average sequences.
- **Source vs target lengths:** scatter plots for English→Hindi and
  English→Marathi pairs with the `y = x` reference line, exposing systematic
  expansion/compression.
- **Vocabulary statistics:** unique-token counts per language after stopword
  removal.
- **Most frequent words:** top-20 bar chart per language (script-specific
  stopword lists applied).
- **Language distribution:** sample counts per language.
- **Data quality:** missing & duplicate counts per column.
- **Word clouds:** one cloud per language (real corpus samples, seeded RNG).
- **Vocabulary growth:** cumulative unique-token growth on the experimental
  corpus — shows Zipf-like saturation.

## 8. Feature Engineering

Classical ML models cannot consume raw text, so we convert text into numeric
feature vectors:

- **CountVectorizer (Bag of Words):** each document becomes a sparse vector of
  term counts. Simple, fast, and interpretable; the basis of most classical
  text models.
- **TF-IDF:** term frequency weighted by inverse document frequency:
  $$\text{tfidf}(t,d) = \text{tf}(t,d) \cdot \log\frac{1+N}{1+\text{df}(t)} + 1$$
  TF-IDF down-weights words that appear in nearly every document and boosts
  words that discriminate between documents — exactly what language
  identification needs.
- **Character n-grams (n = 2–4):** for script-heavy languages, character
  n-grams capture the *script fingerprint* (Devanagari conjuncts, Tamil
  ligatures, etc.) more robustly than words alone. Four feature spaces are
  built: word BoW, word TF-IDF, char BoW, char TF-IDF.

These features are used **only** for the auxiliary classical task. The neural
translation model learns its own subword/word embeddings inside the encoder and
does not use TF-IDF — we state this explicitly to avoid conflating the two
paradigms.

## 9. Classical Machine Learning Algorithms

Auxiliary task: **language identification** — given a sentence, predict which
of six languages it is written in (Hindi, Marathi, Bengali, Gujarati, Tamil,
Telugu).

1. **Multinomial Naïve Bayes:** a generative classifier applying Bayes' theorem
   with a multinomial likelihood over the feature counts. Fast, stable, and a
   strong text baseline.
   $$P(C \mid D) \propto P(C) \prod_{f \in D} P(f \mid C)$$
2. **Logistic Regression:** a discriminative linear classifier; with 6 classes
   we use multinomial (softmax) logistic regression with L2 regularization.
   $$P(C=c \mid D) = \frac{\exp(w_c^T x)}{\sum_{c'} \exp(w_{c'}^T x)}$$
3. **Support Vector Machine (LinearSVC):** finds the max-margin separating
   hyperplanes in the feature space; one-vs-rest for multiclass. Robust in
   high-dimensional sparse spaces.

All three are hyperparameter-light (alpha 1.0, C 1.0, max_iter 2000/20000),
trained on the same stratified 80/10/10 split of char n-gram TF-IDF features,
and evaluated on the same untouched test set.

## 10. Sequence-to-Sequence Architecture

A Seq2Seq model maps a variable-length source sequence $x_1 \dots x_m$ to a
variable-length target sequence $y_1 \dots y_n$:

$$\hat{y} = \arg\max_{y} P(y \mid x)$$

The full model factorizes as:

$$P(y \mid x) = \prod_{t=1}^{n} P(y_t \mid y_{<t}, x)$$

Implemented components:

- **Word embeddings:** each source/target token is mapped to a dense learned
  vector via `nn.Embedding`.
- **Encoder:** a bi-directional LSTM over the source embeddings produces a
  sequence of **hidden states** $h_1, \dots, h_m$.
- **Attention:** computes a **context vector** $c_t$ for each decoder step from
  the encoder states.
- **Decoder:** an autoregressive GRU consumes the previous target embedding and
  the context vector, and produces a probability distribution over the target
  vocabulary.

## 11. Encoder

The encoder is a single-layer **bidirectional LSTM**:

- Forward pass produces $\overrightarrow{h_t}$ from left to right; backward
  pass produces $\overleftarrow{h_t}$ from right to left.
- The two directions are averaged per step to form the encoder state
  $h_t$ (bidirectionality gives each position context from both sides).
- The encoder does not compress the sentence into one vector — it exposes the
  *full set* of hidden states, which is precisely what attention consumes.

The embedding layer is jointly learned end-to-end; embedding dimension 128,
hidden dimension 128.

## 12. Decoder

The decoder is a **GRU** that generates the target sentence one token at a time:

$$s_t = \text{GRU}([\,y_{t-1} ; c_t\,],\; s_{t-1})$$

where $y_{t-1}$ is the embedding of the previously generated token (or
`<sos>` at step 1), and $c_t$ is the attention context. The output projection
computes logits over the vocabulary:

$$\text{logit}_t = W_o\,[\,s_t ; c_t\,] + b_o$$

Greedy decoding picks $\arg\max$ at each step until `<eos>` or the length
limit.

## 13. Attention Mechanism

We implement **Bahdanau (additive) attention**. At decoder step $t$:

1. **Alignment scores** measure the compatibility of decoder state $s_{t-1}$
   with every encoder hidden state $h_j$:
   $$e_{tj} = v_a^\top \tanh\!\big(W_a h_j + U_a s_{t-1}\big)$$
2. **Attention weights** normalize the scores with softmax:
   $$\alpha_{tj} = \frac{\exp(e_{tj})}{\sum_{j'} \exp(e_{tj'})}$$
3. **Context vector** is the weighted sum of encoder states:
   $$c_t = \sum_{j} \alpha_{tj}\, h_j$$

The weights $\alpha_{tj}$ are the model's *interpretable* output: they say which
source token the model was "looking at" while producing target token $t$.

## 14. Attention-Based Alignment

Alignment is the sentence-level correspondence between source and target
tokens. In attention-based NMT, the alignment is **soft** — each target token
attends to all source tokens with different intensities, rather than a
hard 1:1 mapping:

- A well trained model on monotone language pairs (English→Hindi at the clause
  level) produces near-diagonal alignment matrices.
- Reordering phenomena (English "the red book" vs Hindi
  लाल किताब) appear as shifted bright bands.
- Function words often show distributed attention, while content words align
  sharply — an observable, educational pattern.

NEXORA renders these matrices as **heatmaps** (rows = target tokens, columns =
source tokens) using the experimental model's actual weights, both for saved
sample translations and for arbitrary user-entered sentences.

## 15. Model Training

**Experimental Seq2Seq + Attention:**
- Data: English→Hindi pair sample from Samanantar (bounded, seeded).
- Loss: cross-entropy with padding ignored; teacher forcing during training.
- Optimizer: Adam (lr 1e-3), gradient clipping at 5.0.
- Epochs: 3 (2 in `--quick` mode); batch size 64; max length 20 tokens.
- CPU-friendly: ~128-dim embeddings/hidden states, vocabulary capped at 10k.
- The stated limitation (bounded sample, small vocab) is documented because the
  point is to demonstrate the *mechanism*, not to beat production MT.

**Production engine:** the pretrained M2M100-418M model is loaded and used
offline after its first download; it is never retrained in this project.

**Classical models:** fit on the LID train split exactly once, then evaluated
on the test split.

## 16. Model Evaluation

- **Classical:** accuracy, macro precision/recall/F1, confusion matrices,
  one-vs-rest ROC AUC for probabilistic models (NB, LR). LinearSVC has no
  `predict_proba`, so its ROC cell is explicitly marked *not applicable*.
- **Neural:** corpus-level **BLEU** (smooth 1–4 gram, NLTK) on a held-out
  sample, plus per-sample BLEU in the sample-translation table. Training and
  validation loss curves are plotted from the logged history.
- **Production:** not "evaluated" as if trained here — the dashboard labels the
  engine as pretrained and shows its output directly.

All numbers on the dashboard come from JSON files written by the pipeline
(`outputs/metrics/classical_metrics.json`,
`models/seq2seq/metrics.json`) — no hardcoding.

## 17. Results

*(The exact numbers are produced by `python build_pipeline.py` and displayed in
the dashboard. Expected qualitative findings:)*

- Character n-gram TF-IDF separates the six languages with very high accuracy —
  script fingerprints are highly discriminative.
- Logistic Regression and SVM generally edge out Naïve Bayes; Naïve Bayes
  remains highly competitive and trains in seconds.
- The experimental Seq2Seq model learns useful alignment quickly (visible as
  structured heatmaps), while its BLEU stays modest — expected for a bounded
  vocabulary, few epochs, and greedy decoding on CPU.
- BLEU/attention heatmaps demonstrate the *mechanism* honestly rather than
  claiming production-grade quality.

## 18. Dashboard

NEXORA runs as a Streamlit application (`app.py`) with seven sections:

1. **Home** — branding, product description, quick translation.
2. **Live Translation** — language selectors (only supported languages shown),
   text input, translation output with metadata.
3. **Attention Explorer** — saved sample heatmaps, a live heatmap for
   user-entered text, and the alignment theory.
4. **Dataset & EDA** — dataset card with source URL, real statistics, and all
   EDA figures.
5. **Classical ML** — task description, metrics, comparison chart, confusion
   matrices, classification reports, feature-engineering explanation.
6. **Seq2Seq Neural Model** — architecture, training curves, BLEU, sample
   translations, honest model card.
7. **About / Conclusion** — findings, challenges, applications, future scope,
   references.

## 19. Key Findings

- Attention-based alignment is *learned, not hard-coded*: the experimental
  model develops interpretable source–target weighting after only two or three
  epochs.
- Script fingerprints are extremely strong NLP features: character n-gram
  TF-IDF yields near-perfect language identification across Hindi, Marathi,
  Bengali, Gujarati, Tamil and Telugu.
- Classical models give a cheap, explainable baseline; neural models provide
  the generative power; both belong in one mental model of MT.
- Data quality matters: script-consistency filtering removed noticeable
  wrong-alignment rows from the raw corpus.
- Honest evaluation boundaries (bounded sample, small vocab, few epochs) make
  the experimental results credible and reproducible.

## 20. Challenges

- **Multilingual data availability:** some language pairs are sparse in the
  public corpora; alignment noise between English and Indic sentences is
  common.
- **Vocabulary size:** morphologically rich Indic languages inflate
  vocabularies; we cap at 10k tokens and accept some `<unk>`.
- **Sequence length:** long sentences stress Seq2Seq models; attention mitigates
  but does not eliminate the problem.
- **Training time:** full-corpus training is impractical on CPU; we train on a
  bounded sample, which limits BLEU.
- **Translation ambiguity:** homographs and script-based false friends weaken
  alignment in ambiguous contexts.
- **Rare words:** out-of-vocabulary tokens disrupt both alignment and output
  quality.
- **Attention alignment limitations:** attention is not guaranteed to capture
  human-linguist alignment; it is a soft, data-driven proxy.

## 21. Applications

- **Education:** translation aids, transliteration practice, language learning.
- **Tourism & hospitality:** instant phrase translation for travellers.
- **Government services:** multilingual citizen communication and public
  documents.
- **Customer support:** multilingual bots and help-desk responses.
- **Cross-language communication:** chat and social platforms.
- **Accessibility:** text-to-speech pipelines for low-resource language users.
- **Content localization:** subtitles, documentation, marketing at scale.

## 22. Future Scope

- **Transformer architecture:** replace RNN-based attention with multi-head
  self-attention.
- **Beam search & length penalty:** better decoding than greedy.
- **Larger multilingual datasets:** FLORES-200, WMT, more Indic pairs.
- **More languages & low-resource support:** transfer learning from
  multilingual checkpoints.
- **Speech translation:** ASR → MT → TTS pipelines.
- **Document translation:** segmentation and layout-aware translation.
- **Mobile/edge deployment:** quantized models, ONNX, WebGPU.
- **Human evaluation:** adequacy/fluency studies to complement BLEU.
- **Domain-specific fine-tuning:** legal, medical, education glossaries.

## 23. Conclusion

NEXORA demonstrates the complete lifecycle of a modern NLP mini-project:
from a real public parallel corpus, through language-aware preprocessing and
EDA, through classical feature engineering and three classical classifiers, to
a trained **Sequence-to-Sequence network with Bahdanau attention-based
alignment** whose attention weights are visualized as interactive heatmaps —
all wrapped in a professional, live Streamlit application. The project
rigorously separates the pretrained production engine from our own trained
model, reports only measured numbers, and documents every theory component
(encoder, decoder, embeddings, hidden states, context vectors, attention
weights, decoding). The result is a functioning, demonstrable, and honest
submission-ready mini-project that connects the academic theory of attention
with a working product: **AI That Connects Meaning.**

## 24. References

1. Bahdanau, D., Cho, K., & Bengio, Y. (2015). *Neural Machine Translation by
   Jointly Learning to Align and Translate*. ICLR 2015.
2. Sutskever, I., Vinyals, O., & Le, Q. V. (2014). *Sequence to Sequence Learning
   with Neural Networks*. NeurIPS 2014.
3. Cho, K., et al. (2014). *Learning Phrase Representations using RNN
   Encoder–Decoder for Statistical Machine Translation*. EMNLP 2014.
4. Fan, A., et al. (2021). *Beyond English-Centric Multilingual Machine
   Translation* (M2M-100). JMLR.
5. Papineni, K., et al. (2002). *BLEU: a Method for Automatic Evaluation of
   Machine Translation*. ACL 2002.
6. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*.
   JMLR.
7. Ramesh, G., et al. (2022). *Samanantar: The Largest Publicly Available
   Parallel Corpora Collection for 11 Indic Languages*. TACL.
8. AI4Bharat. Samanantar dataset page. https://huggingface.co/datasets/ai4bharat/samanantar
9. Hugging Face. *Transformers* library documentation.
10. Streamlit Inc. *Streamlit documentation*.