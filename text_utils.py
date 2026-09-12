"""
text_utils.py
Shared text-normalization logic used by BOTH data_gen.py (training data) and
predict.py (live inference) so the two stay perfectly in sync. Handles:
- lowercasing
- contraction expansion (also helps negation: "isn't" -> "is not" gives the
  model a clean, separate "not" token instead of one fused word)
- collapsing repeated letters ("sooo" -> "soo")
- light punctuation cleanup while keeping apostrophes (for contractions) and
  sentence-ending punctuation like ! and ? which carry emotional signal
"""

import re

CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "won't": "will not", "wouldn't": "would not",
    "can't": "cannot", "couldn't": "could not", "shouldn't": "should not",
    "mustn't": "must not", "i'm": "i am", "you're": "you are",
    "he's": "he is", "she's": "she is", "it's": "it is", "we're": "we are",
    "they're": "they are", "i've": "i have", "you've": "you have",
    "we've": "we have", "they've": "they have", "i'll": "i will",
    "you'll": "you will", "he'll": "he will", "she'll": "she will",
    "we'll": "we will", "they'll": "they will", "i'd": "i would",
    "you'd": "you would", "he'd": "he would", "she'd": "she would",
    "we'd": "we would", "they'd": "they would", "let's": "let us",
    "that's": "that is", "who's": "who is", "what's": "what is",
    "here's": "here is", "there's": "there is",
}

# common shorthand/typos worth normalizing directly (subword tokenizer
# handles most unseen spellings gracefully, but these are frequent enough
# to fix explicitly)
SHORTHAND = {
    "u": "you", "ur": "your", "r": "are", "luv": "love", "thx": "thanks",
    "pls": "please", "plz": "please", "gr8": "great", "b4": "before",
    "idk": "i do not know", "im": "i am",
}

_repeat_re = re.compile(r"(.)\1{2,}")
_word_re = re.compile(r"[a-z']+|[!?.]")


def normalize_text(text: str) -> str:
    text = text.lower().strip()

    # expand contractions (longest keys first to avoid partial overlaps)
    for contraction in sorted(CONTRACTIONS, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(contraction)}\b", CONTRACTIONS[contraction], text)

    # collapse repeated letters: "sooo" -> "soo" (keeps emphasis signal,
    # removes near-infinite variation)
    text = _repeat_re.sub(r"\1\1", text)

    # keep only letters/apostrophes/basic punctuation, tokenize words + !/?/.
    tokens = _word_re.findall(text)
    tokens = [SHORTHAND.get(tok, tok) for tok in tokens]

    return " ".join(tokens)
