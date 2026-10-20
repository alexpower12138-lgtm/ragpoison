"""Tokenisation shared by retrieval, signals and the data validators.

One tokenizer for the whole project: if the validator ranks documents with a
different tokenizer than the retriever, the ranking audit becomes meaningless.
"""

from __future__ import annotations

import re
import string

#: English stopword list, deliberately short. The default configuration keeps
#: stopwords (``retrieval.drop_stopwords: false``) because the queries are short
#: and BM25 saturates quickly on long ones.
STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for
    with by from as is are was were be been being it its it's their there here
    which who whom whose what when where why how not no nor so such do does did
    doing have has had having will would shall should can could may might must
    about above after again against all am any because before below between
    both during each few further into more most other out over own same some
    too under until up very we you your he she they them his her
    """.split()
)

_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Lowercase and collapse whitespace punctuation into single spaces."""
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


def tokenize(
    text: str,
    lowercase: bool = True,
    strip_punct: bool = True,
    drop_stopwords: bool = False,
) -> list[str]:
    """Split ``text`` into retrieval tokens.

    Numbers are kept (``1987`` stays one token) because many answers in this
    dataset are years, and because the attack families differ in how they
    repeat numbers.
    """
    if not strip_punct:
        raw = re.findall(r"[\w']+", text)
    else:
        raw = _TOKEN_RE.findall(text if not lowercase else text.lower())
    if lowercase:
        tokens = [token.lower() for token in raw]
    else:
        tokens = raw
    if drop_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return tokens


def word_count(text: str) -> int:
    """Whitespace word count, the unit used by the passage-length constraint."""
    return len(text.split())
