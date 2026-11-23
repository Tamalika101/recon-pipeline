"""Description cleaning and similarity.

Bank text and ledger memos use different vocabularies ("ACH DEBIT SYSCO" vs
"Bill payment - Sysco BILL-561"), and bank text is often compressed
("ACHDBTSYSC*X7Q2"). So we compare consonant skeletons of the meaningful
tokens, and allow a partial match of the shorter string inside the longer one.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

VOWELS = set("AEIOU")

# Words that carry no identity: payment rails, document types, legal suffixes.
NOISE_WORDS = {
    "ACH", "DEBIT", "CREDIT", "POS", "WIRE", "IN", "OUT", "DEP", "DEPOSIT", "MOBILE",
    "ONLINE", "PMT", "PAYMENT", "RECEIVED", "BILL", "INV", "INVOICE", "CHECK", "CHK",
    "INC", "LLC", "CO", "THE", "AND", "SUBSCRIPTION", "SETTLEMENT", "TRANSFER",
    "RECURRING", "CHG", "ITEMS", "NET", "PAY", "COM",
}


def skeleton(word: str) -> str:
    """First letter plus consonants: 'PAYMENT' -> 'PYMNT'. Survives vowel-dropping garbling."""
    if not word:
        return ""
    return word[0] + "".join(c for c in word[1:] if c not in VOWELS)


NOISE_SKELETONS = {skeleton(w) for w in NOISE_WORDS}


def tokens(text: str) -> list[str]:
    """Uppercase alphabetic tokens; anything containing a digit is a code and is dropped."""
    raw = re.split(r"[^A-Z0-9]+", text.upper())
    return [t for t in raw if t and not any(ch.isdigit() for ch in t)]


def clean(text: str) -> str:
    """Readable cleaned description: uppercase, codes and noise words removed."""
    kept = [t for t in tokens(text) if t not in NOISE_WORDS]
    return " ".join(kept)


def _is_noise(t: str, s: str) -> bool:
    if t in NOISE_WORDS or s in NOISE_SKELETONS:
        return True
    # Truncated noise word, e.g. "STTL" cut from "SETTLEMENT" at the column limit.
    return len(s) >= 3 and any(n.startswith(s) for n in NOISE_SKELETONS if len(n) > len(s))


def _strip_noise_prefix(s: str) -> str:
    """'PSDBTSLCK' (from 'POSDEBITSLACK' with spaces squeezed out) -> 'SLCK'."""
    changed = True
    while changed:
        changed = False
        for n in sorted(NOISE_SKELETONS, key=len, reverse=True):
            if len(n) >= 2 and s.startswith(n) and len(s) - len(n) >= 3:
                s = s[len(n):]
                changed = True
                break
    return s


def skeleton_tokens(text: str) -> list[str]:
    out = []
    for t in tokens(text):
        s = skeleton(t)
        if len(s) < 2 or _is_noise(t, s):
            continue
        out.append(_strip_noise_prefix(s))
    return out


def _prefix_score(ta: list[str], tb: list[str]) -> float:
    """Share of the shorter token list that is a prefix of a token on the other side.

    Banks truncate names at a column limit ('FD' for 'FedEx'), so a token
    that is the start of the other side's token is real, if weaker, evidence.
    """
    short, long = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    total = 0.0
    for s in short:
        best = 0.0
        for l in long:
            if s == l:
                best = 1.0
            elif l.startswith(s) or s.startswith(l):
                shorter = min(len(s), len(l))
                best = max(best, 1.0 if shorter >= 3 else 0.75)
        total += best
    return total / len(short)


def _partial(short: str, long: str) -> float:
    """Share of the shorter string found in the longer one, in runs of 2+ chars."""
    if not short:
        return 0.0
    blocks = SequenceMatcher(None, short, long, autojunk=False).get_matching_blocks()
    covered = sum(b.size for b in blocks if b.size >= 2)
    score = covered / len(short)
    if len(short) < 4:          # very short names match by accident too easily
        score *= 0.75
    return score


def similarity(a: str, b: str) -> float:
    """0..1 similarity between two descriptions."""
    ta, tb = skeleton_tokens(a), skeleton_tokens(b)
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    jaccard = len(sa & sb) / len(sa | sb)
    ja, jb = "".join(ta), "".join(tb)
    short, long = (ja, jb) if len(ja) <= len(jb) else (jb, ja)
    return max(jaccard, _partial(short, long), _prefix_score(ta, tb))
