"""
NEXORA — Experimental Sequence-to-Sequence model with Bahdanau (additive)
attention-based alignment.

This component is the *academic core* of the project. It is a small, fully
trainable encoder–decoder network that we train on a bounded English→Hindi
sample. It deliberately uses a small vocabulary and a beam-less greedy
decoder so that training stays feasible on CPU while still demonstrating:

  * word embeddings
  * encoder hidden states         h_t = LSTM(x_t, h_{t-1})
  * attention scores              e_tj = v_aᵀ tanh(W_a h_j + U_a s_{t-1})
  * attention weights             α_tj = softmax(e_t)
  * context vectors               c_t  = Σ_j α_tj h_j
  * decoder hidden state          s_t  = LSTM([y_{t-1}, c_t], s_{t-1})
  * output projection + token generation

Production translation in the app is handled by the pretrained
``facebook/m2m100_418M`` model; this file only powers the *experimental*
module and the attention heatmap demos. We never claim this small model was
trained on the full 2M-row corpus.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.multilingual import light_tokenize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS, EOS, PAD, UNK = "<sos>", "<eos>", "<pad>", "<unk>"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocab:
    """Small fixed-size vocabulary with special tokens and Subword-free tokens."""

    def __init__(self, min_freq: int = 2, max_size: int = 8_000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.stoi = {SOS: 0, EOS: 1, PAD: 2, UNK: 3}
        self.itos = [SOS, EOS, PAD, UNK]

    def build(self, sentences) -> None:
        counts = Counter()
        for s in sentences:
            counts.update(light_tokenize(s))
        tokens = [t for t, c in counts.most_common() if c >= self.min_freq]
        for t in tokens[: self.max_size]:
            self.stoi[t] = len(self.itos)
            self.itos.append(t)

    def __len__(self):
        return len(self.itos)

    def encode(self, sentence, max_len: int) -> list[int]:
        toks = light_tokenize(sentence)[: max_len - 2]
        return [self.stoi.get(t, self.stoi[UNK]) for t in toks]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            t = self.itos[i]
            if t in (SOS, EOS, PAD):
                continue
            out.append(t)
        return " ".join(out)

    def tensorize(self, sentences, max_len: int) -> torch.Tensor:
        seqs = []
        for s in sentences:
            ids = [self.stoi[SOS]] + self.encode(s, max_len) + [self.stoi[EOS]]
            seqs.append(ids)
        slen = max(len(s) for s in seqs)
        seqs = [s + [self.stoi[PAD]] * (slen - len(s)) for s in seqs]
        return torch.tensor(seqs, dtype=torch.long, device=DEVICE)


def make_batches(src_texts, tgt_texts, vocab_src, vocab_tgt, batch_size: int, max_len: int, shuffle_rng: random.Random):
    """Yields (src_tensor, tgt_tensor_input, tgt_tensor_target) batches."""
    order = list(range(len(src_texts)))
    shuffle_rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        src_ids = [[vocab_src.stoi[SOS]] + vocab_src.encode(src_texts[i], max_len) + [vocab_src.stoi[EOS]]
                   for i in idx]
        src_len = max(len(s) for s in src_ids)
        src_ids = [s + [vocab_src.stoi[PAD]] * (src_len - len(s)) for s in src_ids]
        src = torch.tensor(src_ids, dtype=torch.long, device=DEVICE)
        tgt_inputs, tgt_targets = [], []
        for i in idx:
            ids = [vocab_tgt.stoi[SOS]] + vocab_tgt.encode(tgt_texts[i], max_len) + [vocab_tgt.stoi[EOS]]
            tgt_inputs.append(ids[:-1])
            tgt_targets.append(ids[1:])
        max_tlen = max(len(t) for t in tgt_inputs)
        tgt_inputs = [t + [vocab_tgt.stoi[PAD]] * (max_tlen - len(t)) for t in tgt_inputs]
        tgt_targets = [t + [vocab_tgt.stoi[PAD]] * (max_tlen - len(t)) for t in tgt_targets]
        yield (src,
               torch.tensor(tgt_inputs, dtype=torch.long, device=DEVICE),
               torch.tensor(tgt_targets, dtype=torch.long, device=DEVICE))


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Bi-directional single-layer LSTM encoder producing hidden states."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):  # src: (B, T)
        emb = self.dropout(self.embed(src))          # (B, T, E)
        hiddens, (h, c) = self.lstm(emb)             # hiddens (B, T, 2*H)
        # combine the two directions into a single hidden_dim state per step
        B, T, _ = hiddens.shape
        hiddens = hiddens.view(B, T, 2, -1).mean(dim=2)   # (B, T, H)
        return hiddens


def _pad_weights(weights, src_len, tgt_len):
    """Pad attention weight matrix to rectangular shape for batching."""
    out = torch.zeros(weights.shape[0], tgt_len, src_len, device=weights.device)
    out[:, : weights.shape[1], : weights.shape[2]] = weights
    return out


# ---------------------------------------------------------------------------
# Attention (Bahdanau, additive)
# ---------------------------------------------------------------------------

class BahdanauAttention(nn.Module):
    r"""
    Bahdanau (additive) attention:

        e_{tj} = v_a^T tanh( W_a h_j + U_a s_{t-1} )     (alignment score)
        α_{tj} = softmax_j( e_{tj} )                     (attention weight)
        c_t    = Σ_j α_{tj} h_j                          (context vector)

    Exposes ``last_weights`` so the dashboard can render real alignment
    heatmaps from this experimental model.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.Wa = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Ua = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.va = nn.Linear(hidden_dim, 1, bias=False)
        self.last_weights: np.ndarray | None = None

    def forward(self, decoder_hidden, encoder_hiddens, src_mask=None):
        # decoder_hidden: (B, H); encoder_hiddens: (B, T, H)
        scored = torch.tanh(self.Wa(encoder_hiddens) + self.Ua(decoder_hidden).unsqueeze(1))
        scores = self.va(scored).squeeze(-1)             # (B, T)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)              # (B, T)
        if src_mask is not None:
            weights = weights.masked_fill(src_mask == 0, 0.0)
        context = torch.bmm(weights.unsqueeze(1), encoder_hiddens).squeeze(1)  # (B, H)
        self.last_weights = weights.detach().cpu().numpy()
        return context, weights


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Gated recurrent unit (GRU) decoder conditioned on the attention context:

        y_t   = Embedding(y_{t-1})
        s_t   = GRU( [ y_t ; c_t ], s_{t-1} )
        logit = W_o ( [ s_t ; c_t ] ) + b_o
        y_t   = argmax softmax(logit)
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.gru = nn.GRU(embed_dim + hidden_dim, hidden_dim, batch_first=True)
        self.attention = BahdanauAttention(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_inputs, encoder_hiddens, src_mask=None, teacher_forcing=True):
        B = tgt_inputs.shape[0]
        s = torch.zeros(1, B, self.gru.hidden_size, device=DEVICE)
        outputs = []
        for t in range(tgt_inputs.shape[1]):
            if t == 0:
                y = self.embed(tgt_inputs[:, 0])          # SOS token
            elif teacher_forcing:
                y = self.embed(tgt_inputs[:, t])
            else:
                y = self.embed(pred_tokens)
            ctx, weights = self.attention(s[-1], encoder_hiddens, src_mask)
            rnn_in = torch.cat([y, ctx], dim=-1).unsqueeze(1)   # (B, 1, E+H)
            out, s = self.gru(rnn_in, s)                        # (B, 1, H), (1,B,H)
            logits = self.fc_out(torch.cat([out.squeeze(1), ctx], dim=-1))
            outputs.append(logits)
            if not teacher_forcing:
                pred_tokens = logits.argmax(dim=-1)
        return torch.stack(outputs, dim=1)  # (B, T, vocab)


# ---------------------------------------------------------------------------
# Sequence-to-Sequence model
# ---------------------------------------------------------------------------

class Seq2SeqAttention(nn.Module):
    """Encoder–Decoder with additive attention-based alignment."""

    def __init__(self, src_vocab_size, tgt_vocab_size, embed_dim=128, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, embed_dim, hidden_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, embed_dim, hidden_dim, dropout)
        self.hidden_dim = hidden_dim

    def forward(self, src, tgt_inputs, src_mask=None, teacher_forcing=True):
        hiddens = self.encoder(src)
        return self.decoder(tgt_inputs, hiddens, src_mask, teacher_forcing)

    @torch.no_grad()
    def translate(self, src_texts: list[str], vocab_src: Vocab, vocab_tgt: Vocab,
                  max_len: int = 30, beam_width: int | None = None):
        """
        Greedy (or beam) decoding of one or more source sentences.
        Returns (predictions, attention_weights) where attention_weights has
        shape (B, tgt_len, src_len).
        """
        self.eval()
        src = vocab_src.tensorize(src_texts, max_len)
        src_mask = (src != PAD_IDX).long()
        hiddens = self.encoder(src)                      # (B, T, H)
        B = src.shape[0]
        s = torch.zeros(1, B, self.hidden_dim, device=DEVICE)
        preds, weights_list = [], []
        tgt_sos = torch.full((B,), vocab_tgt.stoi[SOS], dtype=torch.long, device=DEVICE)
        cur = tgt_sos
        for _ in range(max_len):
            ctx, weights = self.decoder.attention(s[-1], hiddens, src_mask)
            y = self.decoder.embed(cur)
            rnn_in = torch.cat([y, ctx], dim=-1).unsqueeze(1)
            out, s = self.decoder.gru(rnn_in, s)
            logits = self.decoder.fc_out(torch.cat([out.squeeze(1), ctx], dim=-1))
            pred = logits.argmax(dim=-1)
            preds.append(pred)
            weights_list.append(weights)
            cur = pred
        preds = torch.stack(preds, dim=1)                # (B, T)
        weights = torch.stack(weights_list, dim=1)       # (B, T, T_src)
        out = [[vocab_tgt.itos[i] for i in row.cpu().tolist()] for row in preds]
        final = []
        for row in out:
            sent = []
            for t in row:
                if t in (SOS, PAD):
                    continue
                if t == EOS:
                    break
                sent.append(t)
            final.append(" ".join(sent))
        return final, weights.numpy()


PAD_IDX = 2  # must match Vocab order (SOS=0, EOS=1, PAD=2, UNK=3)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_model(
    pair_df,
    out_dir: Path,
    max_len: int = 20,
    batch_size: int = 64,
    epochs: int = 3,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    sample: int = 30_000,
    seed: int = 42,
):
    """Train the experimental Seq2Seq+Attention model on a bounded sample."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    df = pair_df.sample(n=min(sample, len(pair_df)), random_state=seed).reset_index(drop=True)

    vocab_src = Vocab(min_freq=2, max_size=10_000)
    vocab_src.build(df["src"].tolist())
    vocab_tgt = Vocab(min_freq=2, max_size=10_000)
    vocab_tgt.build(df["tgt"].tolist())

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Seq2SeqAttention(len(vocab_src), len(vocab_tgt), embed_dim, hidden_dim).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    n = len(df)
    n_tr, n_va = int(0.85 * n), int(0.1 * n)
    tr = df.iloc[:n_tr].reset_index(drop=True)
    va = df.iloc[n_tr : n_tr + n_va].reset_index(drop=True)

    history = {"epoch": [], "train_loss": [], "valid_loss": []}
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_batches = 0.0, 0
        for src_b, tgt_in, tgt_tg in make_batches(tr["src"].tolist(), tr["tgt"].tolist(),
                                                  vocab_src, vocab_tgt, batch_size, max_len, rng):
            # tensors are already on the target device via Vocab.tensorize
            src_mask = (src_b != PAD_IDX).long()
            logits = model(src_b, tgt_in, src_mask, teacher_forcing=True)
            loss = loss_fn(logits.reshape(-1, len(vocab_tgt)), tgt_tg.reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total += loss.item()
            n_batches += 1
            if n_batches % 25 == 0:
                print(f"  epoch {epoch} batch {n_batches} loss {loss.item():.3f}", flush=True)
        train_loss = total / max(n_batches, 1)

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for src_b, tgt_in, tgt_tg in make_batches(va["src"].tolist(), va["tgt"].tolist(),
                                                      vocab_src, vocab_tgt, batch_size, max_len, rng):
                src_mask = (src_b != PAD_IDX).long()
                logits = model(src_b, tgt_in, src_mask, teacher_forcing=True)
                loss = loss_fn(logits.reshape(-1, len(vocab_tgt)), tgt_tg.reshape(-1))
                vloss += loss.item()
                vn += 1
        valid_loss = vloss / max(vn, 1)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)
        print(f"epoch {epoch}: train {train_loss:.3f} valid {valid_loss:.3f}", flush=True)

    # --- evaluation on a held-out sample -----------------------------------
    te = df.iloc[n_tr + n_va : n_tr + n_va + 200].reset_index(drop=True)
    srcs, refs = te["src"].tolist(), te["tgt"].tolist()
    preds, attn = model.translate(srcs, vocab_src, vocab_tgt, max_len=max_len)

    from src.evaluation import compute_bleu
    bleu = compute_bleu(preds, refs, srcs)

    torch.save(model.state_dict(), out_dir / "seq2seq_attention.pt")
    with open(out_dir / "vocab_src.json", "w", encoding="utf-8") as fh:
        json.dump({"stoi": vocab_src.stoi, "itos": vocab_src.itos}, fh, ensure_ascii=False)
    with open(out_dir / "vocab_tgt.json", "w", encoding="utf-8") as fh:
        json.dump({"stoi": vocab_tgt.stoi, "itos": vocab_tgt.itos}, fh, ensure_ascii=False)
    meta = {
        "epochs": epochs, "batch_size": batch_size, "max_len": max_len,
        "embed_dim": embed_dim, "hidden_dim": hidden_dim, "lr": lr,
        "vocab_src_size": len(vocab_src), "vocab_tgt_size": len(vocab_tgt),
        "samples_trained": n_tr, "samples_validated": n_va,
        "history": history,
        "bleu": bleu,
        "device": str(DEVICE),
        "architecture": "BiLSTM Encoder + GRU Decoder + Bahdanau Additive Attention",
        "note": "Experimental model for academic demonstration. "
                "Production translation uses facebook/m2m100_418M.",
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    sample_rows = [
        {"source": s, "reference": r, "prediction": p}
        for s, r, p in zip(srcs[:10], refs[:10], preds[:10])
    ]
    with open(out_dir / "sample_translations.json", "w", encoding="utf-8") as fh:
        json.dump(sample_rows, fh, indent=2, ensure_ascii=False)
    np.save(out_dir / "attention_sample.npy", attn[:10])
    print(f"[seq2seq] wrote model + metrics to {out_dir}")
    return meta, model, vocab_src, vocab_tgt