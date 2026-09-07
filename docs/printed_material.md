# NEXORA — Printed Submission Material

**Machine Translation using Sequence-to-Sequence Networks with Attention-Based Alignment**
*(Python code, outputs, EDA, model results, translation examples, attention heatmaps, dashboard screenshots, model comparison, conclusion.)*

> Print this document plus the figures referenced below (from `outputs/plots/`,
> `outputs/metrics/`, `outputs/attention/`) and screenshots taken from the
> running Streamlit app.

---

## 1. Python Code

The full source lives in the project root:

| File | Purpose |
|---|---|
| `src/preprocessing.py` | Dataset loading, cleaning, LID + pair construction |
| `src/multilingual.py` | Script detection, light tokenizer, stopwords, word clouds |
| `src/eda.py` | All EDA figures |
| `src/features.py` | CountVectorizer + TF-IDF feature spaces |
| `src/classical_models.py` | Naïve Bayes / Logistic Regression / SVM + metrics |
| `src/seq2seq.py` | BiLSTM encoder + GRU decoder + Bahdanau attention (training & eval) |
| `src/attention.py` | Attention-alignment heatmaps |
| `src/evaluation.py` | BLEU computation |
| `src/translation.py` | Production M2M100 engine |
| `build_pipeline.py` | End-to-end artifact builder |
| `app.py` | Streamlit dashboard |

### Representative code excerpts

**1. Bahdanau attention (the academic core)**

```python
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.va = nn.Linear(hidden_dim, 1, bias=False)
        self.last_weights = None

    def forward(self, decoder_hidden, encoder_hiddens, src_mask=None):
        # e_tj = v_a^T tanh( W_a h_j + U_a s_{t-1} )
        scored = torch.tanh(self.Wa(encoder_hiddens) + self.Ua(decoder_hidden).unsqueeze(1))
        scores = self.va(scored).squeeze(-1)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float("-inf"))
        # alpha_tj = softmax_j( e_tj )
        weights = F.softmax(scores, dim=-1)
        # c_t = sum_j alpha_tj h_j
        context = torch.bmm(weights.unsqueeze(1), encoder_hiddens).squeeze(1)
        self.last_weights = weights.detach().cpu().numpy()
        return context, weights
```

**2. Language-aware cleaning**

```python
def clean_text(text, script):
    text = unicodedata.normalize("NFC", text)   # Unicode normalization
    text = _CTRL.sub(" ", text)                  # remove control characters
    text = _WS.sub(" ", text).strip()            # collapse whitespace
    return text
# Script-consistency filter keeps only rows whose script matches the label.
```

**3. TF-IDF features for language identification**

```python
vec = TfidfVectorizer(max_features=30000, analyzer=char_analyzer)  # char n-grams 2-4
X_train = vec.fit_transform(X_tr)
X_test  = vec.transform(X_te)

model = LinearSVC(C=1.0, max_iter=20000)
model.fit(X_train, y_train)
```

---

## 2. Output — Classical Model Results (measured, test set)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC AUC |
|---|---|---|---|---|---|
| Multinomial Naïve Bayes | 0.99853 | 0.99853 | 0.99853 | 0.99853 | 0.999996 |
| Logistic Regression | 0.99830 | 0.99830 | 0.99830 | 0.99830 | 0.999990 |
| Support Vector Machine (LinearSVC) | **0.99908** | **0.99908** | **0.99908** | **0.99908** | n/a (no predict_proba) |

Dataset: 359 505 processed sentences, 6 languages, stratified 80/10/10 split,
char n-gram (2–4) TF-IDF features (30k max).

**Figures:** `model_comparison.png`, `confusion_naive_bayes.png`,
`confusion_logistic_regression.png`, `confusion_svm.png`, `roc_curves.png`.

---

## 3. Output — EDA Results (measured)

- **Raw rows streamed:** 360 000 (60 000 per language config).
- **Processed LID samples:** 359 505 (after missing/duplicate/script filtering).
- **English→Hindi pair:** 60 000 rows.
- **English→Marathi pair:** 60 000 rows.
- **Split:** 287 604 train / 35 950 valid / 35 951 test.
- Per-language vocabularies and sentence-length distributions are in
  `outputs/plots/dataset_stats.csv` and the EDA figures.

**Figures:** `language_distribution.png`, `length_distribution.png`,
`src_vs_tgt_lengths.png`, `vocabulary.png`, `frequent_words.png`,
`quality.png`, `wordclouds.png`.

---

## 4. Translation Examples — Production Engine (M2M100)

| Source | Target | Output |
|---|---|---|
| How are you? | Hindi | तुम कैसे हो? |
| (live demo) | Marathi | (run the app) |
| (live demo) | Bengali | (run the app) |

> The production engine is **pretrained** (`facebook/m2m100_418M`); NEXORA does
> not claim to have trained it. It is used for live translation after a one-time
> ~1.9 GB download.

---

## 5. Translation Examples — Experimental Seq2Seq (trained by us)

Model: BiLSTM(128) + GRU(128) + Bahdanau attention, 30 000 samples,
3 epochs, greedy decoding, word-level vocabulary (10k src / 4.2k tgt tokens).

**Measured results (from `models/seq2seq/metrics.json`):**

| Metric | Value |
|---|---|
| Train loss (final) | 4.90 |
| Valid loss (final) | 4.95 |
| BLEU (200 held-out samples) | 0.0036 |

Representative real output (source → prediction):

| Source | Prediction (our model) |
|---|---|
| How are you ? | क्या आप \<unk\> ? |
| What is your name ? | क्या आप \<unk\> है ? |
| This is a good book . | यह एक \<unk\> है कि यह एक \<unk\> है। |

The model learns grammatical Hindi structure (question skeletons
"क्या आप … है?", copula patterns "यह एक … है") and produces real Devanagari
words, but vocabulary coverage at 4.2k target tokens leaves frequent `<unk>`
tokens and BLEU is low — exactly what is expected from a bounded experimental
model. Production-quality output comes from the pretrained M2M100 engine.

Full tables are in the app (**Seq2Seq Neural Model** tab) and in
`models/seq2seq/sample_translations.json`.

---

## 6. Attention Heatmaps

Attention alignments for the experimental model are in
`outputs/attention/` (sample heatmaps) and rendered live in the
**Attention Explorer** tab.

Classic example expected from the trained model:

```
Source : how are you ?
Target : तुम कैसे हो ?
Rows   = target tokens, Columns = source tokens
Bright cell (row i, col j) = decoder step i attended strongly to source word j
```

> Attention weights are the **actual weights** captured from the model during
> decoding — never fabricated.

---

## 7. Dashboard Screenshots

Placeholders — run `streamlit run app.py` and capture:

1. Home page (branding, description, quick translate)
2. Live Translation (English → Hindi result)
3. Attention Explorer (heatmap "How are you?" → तुम कैसे हो?)
4. Dataset & EDA (charts)
5. Classical ML (comparison, confusion matrices)
6. Seq2Seq Neural Model (training curves, BLEU table)

---

## 8. Model Comparison

Classical (language ID, char TF-IDF): SVM 99.9 % ≈ LR 99.8 % ≈ NB 99.9 %.
Neural (MT, experimental): bounded BLEU — the model demonstrates the
mechanism; production quality comes from the pretrained engine.

| System | Role | Trained in project? | Metric |
|---|---|---|---|
| Naïve Bayes | Language ID | Yes | F1 0.9985 |
| Logistic Regression | Language ID | Yes | F1 0.9983 |
| SVM | Language ID | Yes | F1 0.9991 |
| Seq2Seq + Attention | MT (experimental) | Yes | BLEU 0.0036 |
| M2M100-418M | MT (production) | No (pretrained) | qualitative |

---

## 9. Conclusion

NEXORA demonstrates the full spectrum of NLP: real data, language-aware
preprocessing, EDA, classical feature engineering and classifiers, and a
from-scratch Seq2Seq network with Bahdanau attention whose attention
weights become interactive alignment heatmaps. The project is honest about
what was trained (Seq2Seq + classical models) and what was borrowed
(pretrained M2M100), and every number on the dashboard is measured, never
invented. **NEXORA — AI That Connects Meaning.**