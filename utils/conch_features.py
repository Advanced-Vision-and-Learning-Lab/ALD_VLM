"""CONCH input contract, including compatibility with current tokenizers."""

import hashlib
from pathlib import Path

import torch.nn.functional as F


FEATURE_VERSION = "conch_v2_official448_tokens128"


def tokenize_conch(tokenizer, texts):
    """CONCH's 127 tokens + reserved CLS placeholder (128 total).

    Matches conch.open_clip_custom.custom_tokenizer.tokenize. Using the
    callable interface also supports transformers versions without batch_encode_plus.
    """
    tokens = tokenizer(
        texts, max_length=127, add_special_tokens=True,
        return_token_type_ids=False, truncation=True, padding="max_length",
        return_tensors="pt",
    )["input_ids"]
    return F.pad(tokens, (0, 1), value=tokenizer.pad_token_id)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
