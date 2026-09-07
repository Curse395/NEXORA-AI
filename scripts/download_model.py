# One-off helper: pre-download the M2M100 production translation model
# so the app loads offline quickly at runtime.
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

model_name = "facebook/m2m100_418M"
print(f"Downloading {model_name} ...", flush=True)
tok = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)
print("M2M100_DOWNLOAD_OK", flush=True)
print("vocab_size", len(tok), flush=True)
print("lang_codes", len(tok.lang_code_to_id), flush=True)