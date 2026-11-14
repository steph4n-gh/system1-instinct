"""Small, portable TF-IDF features learned from teaching text; NumPy only."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Match word unigram/bigram TF-IDF with smoothed IDF and sublinear term frequency.
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
_CONTRACTION_RE = re.compile(
    r"\b(can't|won't|isn't|wasn't|didn't|don't|doesn't|couldn't|shouldn't|wouldn't|aren't|weren't|haven't|hasn't|hadn't)\b"
)
_NEGATION_TOKENS: frozenset[str] = frozenset({
    "not",
    "never",
    "no",
    "without",
    "cant",
    "cannot",
    "wont",
    "isnt",
    "wasnt",
    "didnt",
    "dont",
    "doesnt",
    "couldnt",
    "shouldnt",
    "wouldnt",
    "arent",
    "werent",
    "havent",
    "hasnt",
    "hadnt",
    "neither",
    "nor",
})
_NEGATION_SKIP_MODIFIERS: frozenset[str] = frozenset({
    "the",
    "a",
    "an",
    "to",
    "be",
    "been",
    "being",
    "very",
    "really",
    "so",
    "too",
    "quite",
    "ever",
    "even",
    "yet",
    "still",
    "at",
    "all",
})
DEFAULT_MAX_FEATURES = 4096
MAX_FEATURES = 8192
MAX_TERM_LENGTH = 128
MAX_TEXT_LENGTH = 8192
DEFAULT_CHAR_WEIGHT = 0.32
DEFAULT_STEM_WEIGHT = 0.70
DEFAULT_SKIP_WEIGHT = 0.65
DEFAULT_WORD_QUOTA_FRAC = 0.62


def _stem_word(word: str) -> str:
    """Lightweight deterministic suffix stemmer for optional Wave 2 lexical normalization."""
    if len(word) <= 4:
        return word
    for suffix, min_root in (
        ("ations", 3),
        ("ation", 3),
        ("ments", 4),
        ("ment", 4),
        ("ness", 4),
        ("ings", 3),
        ("ing", 3),
        ("ied", 3),
        ("ies", 3),
        ("able", 4),
        ("ible", 4),
        ("ally", 4),
        ("ed", 3),
        ("es", 3),
        ("ly", 4),
        ("er", 4),
        ("s", 4),
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= min_root:
            stem = word[: -len(suffix)]
            if suffix in ("ied", "ies"):
                stem += "y"
            return stem
    return word


def _validate_char_ngrams(char_ngrams: Sequence[int] | None) -> tuple[int, int] | None:
    if char_ngrams is None:
        return None
    if (
        not isinstance(char_ngrams, (list, tuple))
        or len(char_ngrams) != 2
        or any(type(n) is not int for n in char_ngrams)
    ):
        raise ValueError("char_ngrams must be a (min_n, max_n) integer pair")
    min_n, max_n = int(char_ngrams[0]), int(char_ngrams[1])
    if not (2 <= min_n <= max_n <= 6):
        raise ValueError("char_ngrams must satisfy 2 <= min_n <= max_n <= 6")
    return (min_n, max_n)


_CLAUSE_SPLIT_RE = re.compile(r"[.!?;,]+")


def _terms(
    text: str,
    char_ngrams: tuple[int, int] | None = None,
    *,
    use_stems: bool = False,
    use_skip_bigrams: bool = False,
) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("TF-IDF input must be text")
    lowered = _CONTRACTION_RE.sub(lambda m: m.group(0).replace("'", ""), text[:MAX_TEXT_LENGTH].lower())
    words = _TOKEN.findall(lowered)
    out = [
        term
        for term in words + [f"{a} {b}" for a, b in zip(words, words[1:])]
        if len(term) <= MAX_TERM_LENGTH
    ]
    negated_words: set[str] = set()
    for clause in _CLAUSE_SPLIT_RE.split(lowered):
        c_words = _TOKEN.findall(clause)
        n_c = len(c_words)
        if use_skip_bigrams and n_c >= 3:
            for a, c in zip(c_words, c_words[2:]):
                skip_tok = f"skip:{a}_{c}"
                if len(skip_tok) <= MAX_TERM_LENGTH:
                    out.append(skip_tok)
        for i, w in enumerate(c_words):
            if w in _NEGATION_TOKENS:
                bound: list[str] = []
                for j in range(i + 1, min(i + 4, n_c)):
                    target = c_words[j]
                    if target in _NEGATION_TOKENS:
                        break
                    if target in _NEGATION_SKIP_MODIFIERS:
                        continue
                    bound.append(target)
                    negated_words.add(target)
                    neg_canon = f"neg:{target}"
                    if len(neg_canon) <= MAX_TERM_LENGTH:
                        out.append(neg_canon)
                    neg_tok = f"{w}_{target}"
                    if len(neg_tok) <= MAX_TERM_LENGTH:
                        out.append(neg_tok)
                    if j > i + 1:
                        neg_pair = f"{w} {target}"
                        if len(neg_pair) <= MAX_TERM_LENGTH:
                            out.append(neg_pair)
                    if len(bound) == 2:
                        span_tok = f"{w}_{bound[0]}_{bound[1]}"
                        if len(span_tok) <= MAX_TERM_LENGTH:
                            out.append(span_tok)
                        break
    if use_stems:
        for w in words:
            if w not in negated_words and len(w) >= 4 and len(w) <= MAX_TERM_LENGTH:
                st = _stem_word(w)
                if st != w and len(st) >= 3:
                    st_tok = f"stem:{st}"
                    if len(st_tok) <= MAX_TERM_LENGTH:
                        out.append(st_tok)
    if char_ngrams is not None:
        min_n, max_n = char_ngrams
        for w in words:
            if w not in negated_words and len(w) <= MAX_TERM_LENGTH:
                padded = f"<{w}>"
                plen = len(padded)
                for n in range(min_n, max_n + 1):
                    for i in range(plen - n + 1):
                        out.append(f"#{padded[i:i + n]}")
    return out


def _select_pool_round_robin(
    pool_terms: set[str],
    per_class_positive: Mapping[str, Sequence[tuple[float, int, int, str]]],
    global_scores: Mapping[str, float],
    frequency: Counter[str],
    classes: Sequence[str],
    budget: int,
    quota_override: int | None,
    chosen_set: set[str],
) -> list[str]:
    if budget <= 0 or not pool_terms:
        return []
    quota = (
        quota_override
        if quota_override is not None
        else max(1, budget // max(1, 2 * len(classes)))
    )
    selected: list[str] = []
    if quota > 0 and classes:
        filtered_by_class = {
            c: [item for item in per_class_positive[c] if item[3] in pool_terms]
            for c in classes
        }
        max_len = max((len(v) for v in filtered_by_class.values()), default=0)
        per_class_added = {c: 0 for c in classes}
        for rank_idx in range(max_len):
            if len(selected) >= budget:
                break
            any_added = False
            for c in classes:
                if per_class_added[c] >= quota:
                    continue
                candidates = filtered_by_class[c]
                if rank_idx < len(candidates):
                    term = candidates[rank_idx][3]
                    if term not in chosen_set:
                        selected.append(term)
                        chosen_set.add(term)
                        per_class_added[c] += 1
                        any_added = True
                        if len(selected) >= budget:
                            break
                else:
                    any_added = True
            if not any_added and all(
                per_class_added[c] >= min(quota, len(filtered_by_class[c])) for c in classes
            ):
                break

    if len(selected) < budget:
        remaining = sorted(
            (term for term in pool_terms if term not in chosen_set),
            key=lambda term: (
                -round(global_scores.get(term, 0.0), 12),
                len(term) if term.startswith("#") else 0,
                -frequency[term],
                term,
            ),
        )
        take = remaining[: budget - len(selected)]
        selected.extend(take)
        chosen_set.update(take)
    return selected


def _select_supervised_terms(
    frequency: Counter[str],
    document_frequency: Counter[str],
    term_class_df: Mapping[str, Counter[str]],
    class_counts: Counter[str],
    total_docs: int,
    max_features: int,
    selection: str,
    min_terms_per_class: int | None,
    word_quota_frac: float = DEFAULT_WORD_QUOTA_FRAC,
) -> list[str]:
    classes = sorted(class_counts.keys())
    if len(classes) <= 1 or total_docs <= 1:
        return sorted(
            sorted(
                frequency,
                key=lambda term: (-frequency[term], 1 if term.startswith("#") else 0, term),
            )[:max_features]
        )

    n_total = float(total_docs)
    use_support_weight = total_docs >= 40 and (n_total / len(classes)) >= 3.0
    per_class_positive: dict[str, list[tuple[float, int, int, str]]] = {c: [] for c in classes}
    global_scores: dict[str, float] = {}

    for term, df_int in document_frequency.items():
        df = float(df_int)
        not_df = n_total - df
        if df <= 0.0 or not_df <= 0.0:
            global_scores[term] = 0.0
            continue
        c_dfs = term_class_df[term]
        best_score = 0.0
        len_tie = len(term) if term.startswith("#") else 0
        support_weight = math.log1p(df) if use_support_weight else 1.0
        for c in classes:
            n_c = float(class_counts[c])
            not_n_c = n_total - n_c
            if n_c <= 0.0 or not_n_c <= 0.0:
                continue
            a = float(c_dfs.get(c, 0))
            b = df - a
            c_neg = n_c - a
            d = not_df - c_neg
            expected_a = (df * n_c) / n_total

            if selection == "chi2":
                num = n_total * (a * d - b * c_neg) ** 2
                den = df * not_df * n_c * not_n_c
                score = (num / den if den > 0.0 else 0.0) * support_weight
            else:
                score = 0.0
                for obs, row_sum, col_sum in (
                    (a, df, n_c),
                    (b, df, not_n_c),
                    (c_neg, not_df, n_c),
                    (d, not_df, not_n_c),
                ):
                    if obs > 0.0 and row_sum > 0.0 and col_sum > 0.0:
                        score += (obs / n_total) * math.log((n_total * obs) / (row_sum * col_sum))
                score *= support_weight

            if use_support_weight:
                if df_int == 1:
                    score *= 0.05
                elif a <= 1.0:
                    score *= 0.25

            if score > best_score:
                best_score = score
            if a > expected_a and score > 0.0:
                per_class_positive[c].append((-round(score, 12), len_tie, -frequency[term], term))
        global_scores[term] = best_score

    for c in classes:
        per_class_positive[c].sort()

    word_terms = {t for t in frequency if not t.startswith("#")}
    char_terms = {t for t in frequency if t.startswith("#")}
    chosen: list[str] = []
    chosen_set: set[str] = set()

    if not char_terms:
        chosen.extend(
            _select_pool_round_robin(
                word_terms,
                per_class_positive,
                global_scores,
                frequency,
                classes,
                max_features,
                min_terms_per_class,
                chosen_set,
            )
        )
    else:
        word_budget = min(len(word_terms), max(1, int(max_features * float(word_quota_frac))))
        word_quota = (
            min_terms_per_class
            if min_terms_per_class is not None
            else max(1, word_budget // max(1, int(1.5 * len(classes))))
        )
        chosen.extend(
            _select_pool_round_robin(
                word_terms,
                per_class_positive,
                global_scores,
                frequency,
                classes,
                word_budget,
                word_quota,
                chosen_set,
            )
        )
        char_budget = max_features - len(chosen)
        char_quota = max(1, char_budget // max(1, 2 * len(classes)))
        chosen.extend(
            _select_pool_round_robin(
                char_terms,
                per_class_positive,
                global_scores,
                frequency,
                classes,
                char_budget,
                char_quota,
                chosen_set,
            )
        )
        if len(chosen) < max_features:
            chosen.extend(
                _select_pool_round_robin(
                    word_terms,
                    per_class_positive,
                    global_scores,
                    frequency,
                    classes,
                    max_features - len(chosen),
                    None,
                    chosen_set,
                )
            )

    return sorted(chosen[:max_features])


@dataclass(frozen=True, init=False, repr=False, eq=False)
class TfidfProjector:
    """Frozen word unigram/bigram (and optional subword char n-gram) features fitted only on teaching examples.

    Fit before compiling, then supply separate calibration examples. The vocabulary
    and IDF values travel with the saved skill. New vocabulary requires a new skill;
    ``project`` never updates the feature space or contacts a teacher.
    """

    _terms: tuple[str, ...]
    _idf: tuple[float, ...]
    _char_ngrams: tuple[int, int] | None
    _vocabulary: Mapping[str, int]
    _digest: str
    _use_stems: bool
    _stem_weight: float
    _use_skip_bigrams: bool
    _skip_weight: float
    _word_quota_frac: float
    _char_weight: float

    def __init__(
        self,
        terms: Sequence[str],
        idf: Sequence[float],
        *,
        char_ngrams: Sequence[int] | None = None,
        use_stems: bool = False,
        stem_weight: float = DEFAULT_STEM_WEIGHT,
        use_skip_bigrams: bool = False,
        skip_weight: float = DEFAULT_SKIP_WEIGHT,
        word_quota_frac: float = DEFAULT_WORD_QUOTA_FRAC,
        char_weight: float = DEFAULT_CHAR_WEIGHT,
    ) -> None:
        if not isinstance(terms, (list, tuple)) or not 0 < len(terms) <= MAX_FEATURES:
            raise ValueError(f"TF-IDF vocabulary must contain 1–{MAX_FEATURES} terms")
        if any(not isinstance(term, str) or not term or len(term) > MAX_TERM_LENGTH for term in terms):
            raise ValueError("TF-IDF terms must be bounded nonempty strings")
        if len(set(terms)) != len(terms):
            raise ValueError("TF-IDF terms must be unique")
        if not isinstance(idf, (list, tuple)) or len(idf) != len(terms):
            raise ValueError("TF-IDF weights must match the vocabulary")
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 100 for value in idf):
            raise ValueError("TF-IDF weights must be finite values between 1 and 100")
        for weight_name, weight_val in (
            ("stem_weight", stem_weight),
            ("skip_weight", skip_weight),
            ("char_weight", char_weight),
        ):
            if type(weight_val) not in (int, float) or not math.isfinite(weight_val) or not 0.0 < float(weight_val) <= 5.0:
                raise ValueError(f"{weight_name} must be in (0, 5]")
        if (
            type(word_quota_frac) not in (int, float)
            or not math.isfinite(word_quota_frac)
            or not 0.05 < float(word_quota_frac) < 0.98
        ):
            raise ValueError("word_quota_frac must be in (0.05, 0.98)")

        validated_ngrams = _validate_char_ngrams(char_ngrams)
        vocab = MappingProxyType({term: i for i, term in enumerate(terms)})
        object.__setattr__(self, "_terms", tuple(terms))
        object.__setattr__(self, "_idf", tuple(float(value) for value in idf))
        object.__setattr__(self, "_char_ngrams", validated_ngrams)
        object.__setattr__(self, "_vocabulary", vocab)
        object.__setattr__(self, "_use_stems", bool(use_stems))
        object.__setattr__(self, "_stem_weight", float(stem_weight))
        object.__setattr__(self, "_use_skip_bigrams", bool(use_skip_bigrams))
        object.__setattr__(self, "_skip_weight", float(skip_weight))
        object.__setattr__(self, "_word_quota_frac", float(word_quota_frac))
        object.__setattr__(self, "_char_weight", float(char_weight))
        self._rebuild_fast_tables()
        object.__setattr__(self, "_digest", hashlib.sha256(json.dumps(self.to_config(), sort_keys=True).encode()).hexdigest())

    def _rebuild_fast_tables(self) -> None:
        terms = self._terms
        idf_f64 = np.asarray(self._idf, dtype=np.float64)
        scale_f64 = np.ones(len(terms), dtype=np.float64)
        word_vocab: dict[str, int] = {}
        char_substr_to_idx: dict[str, int] = {}
        bigram_first: set[str] = set()
        neg_targets: dict[str, int] = {}

        for idx, term in enumerate(terms):
            if term.startswith("#"):
                scale_f64[idx] = self._char_weight
                char_substr_to_idx[term[1:]] = idx
            else:
                word_vocab[term] = idx
                if term.startswith("neg:") or term.startswith("not_"):
                    scale_f64[idx] = 1.6
                    if term.startswith("neg:"):
                        neg_targets[term[4:]] = idx
                elif term.startswith("stem:"):
                    scale_f64[idx] = self._stem_weight
                elif term.startswith("skip:"):
                    scale_f64[idx] = self._skip_weight
                if " " in term:
                    bigram_first.add(term.split(" ", 1)[0])

        if self._char_ngrams is not None and char_substr_to_idx:
            min_n, max_n = self._char_ngrams
            char_lens = tuple(sorted({len(k) for k in char_substr_to_idx if min_n <= len(k) <= max_n}))
        else:
            char_lens = ()

        object.__setattr__(self, "_idf_f64", idf_f64)
        object.__setattr__(self, "_scale_f64", scale_f64)
        object.__setattr__(self, "_word_vocab", word_vocab)
        object.__setattr__(self, "_bigram_first_words", frozenset(bigram_first))
        object.__setattr__(self, "_neg_targets_in_vocab", neg_targets)
        object.__setattr__(self, "_char_substr_to_idx", char_substr_to_idx)
        object.__setattr__(self, "_char_lens", char_lens)
        object.__setattr__(self, "_word_char_cache", {})
        object.__setattr__(self, "_neg_suppress_cache", None)

    def _get_neg_suppress_map(self) -> dict[str, tuple[int, ...]]:
        cached = self._neg_suppress_cache
        if cached is None:
            raw_map: dict[str, list[int]] = defaultdict(list)
            for idx, term in enumerate(self._terms):
                if term.startswith("#") or term.startswith("neg:") or term.startswith("not_"):
                    continue
                seen_toks = {term}
                if " " in term:
                    seen_toks.update(term.split())
                for tok in seen_toks:
                    raw_map[tok].append(idx)
            cached = {k: tuple(v) for k, v in raw_map.items()}
            object.__setattr__(self, "_neg_suppress_cache", cached)
        return cached

    def _word_char_hits(self, word: str) -> tuple[int, ...]:
        cache: dict[str, tuple[int, ...]] = self._word_char_cache
        hits = cache.get(word)
        if hits is not None:
            return hits
        padded = f"<{word}>"
        plen = len(padded)
        c_get = self._char_substr_to_idx.get
        found: list[int] = []
        for n in self._char_lens:
            if plen >= n:
                for i in range(plen - n + 1):
                    idx = c_get(padded[i : i + n])
                    if idx is not None:
                        found.append(idx)
        result = tuple(found)
        if len(cache) < 32768:
            cache[word] = result
        return result

    def _extract_matched_counts(self, text: str) -> tuple[dict[int, int], set[int]]:
        if not isinstance(text, str):
            raise TypeError("TF-IDF input must be text")
        lowered = _CONTRACTION_RE.sub(lambda m: m.group(0).replace("'", ""), text[:MAX_TEXT_LENGTH].lower())
        words = _TOKEN.findall(lowered)
        counts: dict[int, int] = {}
        w_counts: dict[str, int] = {}
        w_get = self._word_vocab.get
        bi_first = self._bigram_first_words

        for w in words:
            w_counts[w] = w_counts.get(w, 0) + 1
            idx = w_get(w)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1

        if bi_first and len(words) > 1:
            for i in range(len(words) - 1):
                a = words[i]
                if a in bi_first:
                    idx = w_get(f"{a} {words[i + 1]}")
                    if idx is not None:
                        counts[idx] = counts.get(idx, 0) + 1

        negated_words: set[str] = set()
        has_neg = not _NEGATION_TOKENS.isdisjoint(w_counts)
        if has_neg or self._use_skip_bigrams:
            for clause in _CLAUSE_SPLIT_RE.split(lowered):
                c_words = _TOKEN.findall(clause)
                n_c = len(c_words)
                if self._use_skip_bigrams and n_c >= 3:
                    for a, c in zip(c_words, c_words[2:]):
                        idx = w_get(f"skip:{a}_{c}")
                        if idx is not None:
                            counts[idx] = counts.get(idx, 0) + 1
                if has_neg:
                    for i, w in enumerate(c_words):
                        if w in _NEGATION_TOKENS:
                            bound: list[str] = []
                            for j in range(i + 1, min(i + 4, n_c)):
                                target = c_words[j]
                                if target in _NEGATION_TOKENS:
                                    break
                                if target in _NEGATION_SKIP_MODIFIERS:
                                    continue
                                bound.append(target)
                                negated_words.add(target)
                                for tok in (f"neg:{target}", f"{w}_{target}"):
                                    idx = w_get(tok)
                                    if idx is not None:
                                        counts[idx] = counts.get(idx, 0) + 1
                                if j > i + 1:
                                    idx = w_get(f"{w} {target}")
                                    if idx is not None:
                                        counts[idx] = counts.get(idx, 0) + 1
                                if len(bound) == 2:
                                    idx = w_get(f"{w}_{bound[0]}_{bound[1]}")
                                    if idx is not None:
                                        counts[idx] = counts.get(idx, 0) + 1
                                    break

        if self._use_stems:
            for w, wc in w_counts.items():
                if w not in negated_words and 4 <= len(w) <= MAX_TERM_LENGTH:
                    st = _stem_word(w)
                    if st != w and len(st) >= 3:
                        idx = w_get(f"stem:{st}")
                        if idx is not None:
                            counts[idx] = counts.get(idx, 0) + wc

        if self._char_lens:
            char_hits_fn = self._word_char_hits
            for w, wc in w_counts.items():
                if w not in negated_words and len(w) <= MAX_TERM_LENGTH:
                    for c_idx in char_hits_fn(w):
                        counts[c_idx] = counts.get(c_idx, 0) + wc

        suppressed: set[int] = set()
        if negated_words and self._neg_targets_in_vocab:
            active_tgts = [w for w in negated_words if w in self._neg_targets_in_vocab]
            if active_tgts:
                supp_map = self._get_neg_suppress_map()
                for tgt in active_tgts:
                    s_idxs = supp_map.get(tgt)
                    if s_idxs:
                        suppressed.update(s_idxs)

        return counts, suppressed

    @property
    def dimension(self) -> int:
        return len(self._terms)

    @property
    def char_ngrams(self) -> tuple[int, int] | None:
        return self._char_ngrams

    @property
    def vocabulary(self) -> Mapping[str, int]:
        return self._vocabulary

    def __call__(self, text: str) -> np.ndarray:
        return self.project(text)

    @classmethod
    def fit(
        cls,
        texts: Iterable[str],
        *,
        max_features: int = 2048,
        labels: Sequence[Any] | None = None,
        selection: str = "frequency",
        char_ngrams: Sequence[int] | None = None,
        min_terms_per_class: int | None = None,
        use_stems: bool = False,
        stem_weight: float = DEFAULT_STEM_WEIGHT,
        use_skip_bigrams: bool = False,
        skip_weight: float = DEFAULT_SKIP_WEIGHT,
        word_quota_frac: float = DEFAULT_WORD_QUOTA_FRAC,
        char_weight: float = DEFAULT_CHAR_WEIGHT,
    ) -> TfidfProjector:
        """Choose frequent or class-discriminative terms, retaining deterministic lexicographic tie breaks."""
        if type(max_features) is not int or not 1 <= max_features <= MAX_FEATURES:
            raise ValueError(f"max_features must be an integer between 1 and {MAX_FEATURES}")
        if isinstance(texts, str):
            raise TypeError("Pass a collection of teaching texts")
        if selection not in ("frequency", "chi2", "mi"):
            raise ValueError("selection must be 'frequency', 'chi2', or 'mi'")
        if selection in ("chi2", "mi") and labels is None:
            raise ValueError(f"selection={selection!r} requires teaching labels")
        if min_terms_per_class is not None and (type(min_terms_per_class) is not int or min_terms_per_class < 0):
            raise ValueError("min_terms_per_class must be a nonnegative integer")
        validated_ngrams = _validate_char_ngrams(char_ngrams)

        text_list = list(texts)
        label_list: list[str] | None = None
        if labels is not None:
            if isinstance(labels, (str, bytes)):
                raise TypeError("labels must be a sequence matching texts")
            raw_labels = list(labels)
            if len(raw_labels) != len(text_list):
                raise ValueError("labels length must match texts length")
            label_list = [str(lbl) for lbl in raw_labels]

        frequency: Counter[str] = Counter()
        document_frequency: Counter[str] = Counter()
        term_class_df: dict[str, Counter[str]] = defaultdict(Counter)
        class_counts: Counter[str] = Counter()
        count = 0
        for idx, text in enumerate(text_list):
            terms = _terms(
                text,
                validated_ngrams,
                use_stems=bool(use_stems),
                use_skip_bigrams=bool(use_skip_bigrams),
            )
            frequency.update(terms)
            unique_terms = set(terms)
            document_frequency.update(unique_terms)
            if label_list is not None:
                lbl = label_list[idx]
                class_counts[lbl] += 1
                for t in unique_terms:
                    term_class_df[t][lbl] += 1
            count += 1
        if not frequency:
            raise ValueError("Teaching texts must contain word features")
        if selection == "frequency" or label_list is None:
            selected = sorted(
                sorted(
                    frequency,
                    key=lambda term: (-frequency[term], 1 if term.startswith("#") else 0, term),
                )[:max_features]
            )
        else:
            selected = _select_supervised_terms(
                frequency,
                document_frequency,
                term_class_df,
                class_counts,
                count,
                max_features,
                selection,
                min_terms_per_class,
                word_quota_frac=float(word_quota_frac),
            )
        return cls(
            selected,
            [math.log((1 + count) / (1 + document_frequency[term])) + 1 for term in selected],
            char_ngrams=validated_ngrams,
            use_stems=bool(use_stems),
            stem_weight=float(stem_weight),
            use_skip_bigrams=bool(use_skip_bigrams),
            skip_weight=float(skip_weight),
            word_quota_frac=float(word_quota_frac),
            char_weight=float(char_weight),
        )

    def raw_vector(self, text: str) -> np.ndarray:
        """Compute unnormalized TF-IDF vector for familiarity/support inspection."""
        vector = np.zeros(self.dimension, dtype=np.float32)
        vocab = self._vocabulary
        term_counts = Counter(
            _terms(
                text,
                self._char_ngrams,
                use_stems=self._use_stems,
                use_skip_bigrams=self._use_skip_bigrams,
            )
        )
        matched = {vocab[t]: count for t, count in term_counts.items() if t in vocab}
        if not matched:
            return vector
        neg_targets = self._neg_targets_in_vocab
        suppressed: set[int] = set()
        if neg_targets:
            active_tgts = [
                term[4:]
                for term in term_counts
                if term.startswith("neg:") and term[4:] in neg_targets
            ]
            if active_tgts:
                supp_map = self._get_neg_suppress_map()
                for tgt in active_tgts:
                    s_idxs = supp_map.get(tgt)
                    if s_idxs:
                        suppressed.update(s_idxs)
        idf = self._idf_f64
        scale_arr = self._scale_f64
        for index, count in matched.items():
            scale = 0.15 if index in suppressed else float(scale_arr[index])
            tf = 1.0 if count == 1 else (1.0 + math.log(count))
            vector[index] = scale * tf * float(idf[index])
        return vector

    def raw_norm(self, text: str) -> float:
        """Return pre-normalization L2 norm of matched TF-IDF features."""
        if not isinstance(text, str):
            raise TypeError("TF-IDF input must be text")
        return float(np.linalg.norm(self.raw_vector(text)))

    def project_with_norm(self, text: str, recency_weighted: bool | None = None) -> tuple[np.ndarray, float]:
        """Return unit-normalized TF-IDF vector and pre-normalization L2 norm in one pass."""
        if recency_weighted:
            raise ValueError("TF-IDF uses document frequencies, not recency weighting")
        if not isinstance(text, str):
            raise TypeError("TF-IDF input must be text")
        vector = self.raw_vector(text)
        norm = float(np.linalg.norm(vector))
        return (vector / norm if norm > 0 else vector), norm

    def project(self, text: str, recency_weighted: bool | None = None) -> np.ndarray:
        return self.project_with_norm(text, recency_weighted)[0]

    def project_batch(
        self,
        texts: Sequence[str],
        recency_weighted: bool | None = None,
        *,
        return_norms: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if recency_weighted:
            raise ValueError("TF-IDF uses document frequencies, not recency weighting")
        n_docs = len(texts)
        if n_docs == 0:
            empty_mat = np.empty((0, self.dimension), dtype=np.float32)
            return (empty_mat, np.empty(0, dtype=np.float32)) if return_norms else empty_mat

        out = np.zeros((n_docs, self.dimension), dtype=np.float32)
        rows: list[int] = []
        cols: list[int] = []
        cnts: list[int] = []
        scales: list[float] = []
        scale_arr = self._scale_f64

        for row_idx, text in enumerate(texts):
            counts, suppressed = self._extract_matched_counts(text)
            if not counts:
                continue
            if not suppressed:
                c_keys = list(counts.keys())
                rows.extend([row_idx] * len(c_keys))
                cols.extend(c_keys)
                cnts.extend(counts.values())
                scales.extend(scale_arr[c_keys].tolist())
            else:
                for col_idx, count in counts.items():
                    rows.append(row_idx)
                    cols.append(col_idx)
                    cnts.append(count)
                    scales.append(0.15 if col_idx in suppressed else float(scale_arr[col_idx]))

        if rows:
            r_arr = np.asarray(rows, dtype=np.intp)
            c_arr = np.asarray(cols, dtype=np.intp)
            n_arr = np.asarray(cnts, dtype=np.float64)
            s_arr = np.asarray(scales, dtype=np.float64)
            tf_arr = np.where(n_arr == 1.0, 1.0, 1.0 + np.log(n_arr))
            out[r_arr, c_arr] = (s_arr * tf_arr * self._idf_f64[c_arr]).astype(np.float32)

        norms = np.linalg.norm(out, axis=1)
        nz = norms > 0.0
        if np.any(nz):
            out[nz] = out[nz] / norms[nz, None]
        if return_norms:
            return out, norms
        return out

    def projector_digest(self) -> str:
        return self._digest

    def to_config(self, *, include_arrays: bool = True) -> dict:
        cfg: dict[str, Any] = {"type": "tfidf", "terms": list(self._terms)}
        if include_arrays:
            cfg["idf"] = list(self._idf)
        if self._char_ngrams is not None:
            cfg["char_ngrams"] = list(self._char_ngrams)
        if len(self._terms) > DEFAULT_MAX_FEATURES:
            cfg["max_features"] = len(self._terms)
        if self._use_stems:
            cfg["use_stems"] = True
        if self._stem_weight != DEFAULT_STEM_WEIGHT:
            cfg["stem_weight"] = self._stem_weight
        if self._use_skip_bigrams:
            cfg["use_skip_bigrams"] = True
        if self._skip_weight != DEFAULT_SKIP_WEIGHT:
            cfg["skip_weight"] = self._skip_weight
        if self._word_quota_frac != DEFAULT_WORD_QUOTA_FRAC:
            cfg["word_quota_frac"] = self._word_quota_frac
        if self._char_weight != DEFAULT_CHAR_WEIGHT:
            cfg["char_weight"] = self._char_weight
        return cfg

    @classmethod
    def validate_header_config(
        cls,
        config: dict,
        *,
        allow_missing_idf: bool = False,
    ) -> tuple[list[str], int]:
        allowed_keys = {
            "type",
            "terms",
            "idf",
            "char_ngrams",
            "max_features",
            "use_stems",
            "stem_weight",
            "use_skip_bigrams",
            "skip_weight",
            "word_quota_frac",
            "char_weight",
        }
        if (
            not isinstance(config, dict)
            or not {"type", "terms"} <= set(config)
            or (not allow_missing_idf and "idf" not in config)
            or set(config) - allowed_keys
            or config["type"] != "tfidf"
        ):
            raise ValueError("Unsupported TF-IDF settings")
        limit = DEFAULT_MAX_FEATURES
        if "max_features" in config:
            mf = config["max_features"]
            if type(mf) is not int or not DEFAULT_MAX_FEATURES < mf <= MAX_FEATURES:
                raise ValueError("Unsupported TF-IDF max_features setting")
            limit = mf
        for b_key in ("use_stems", "use_skip_bigrams"):
            if b_key in config and type(config[b_key]) is not bool:
                raise ValueError(f"Unsupported TF-IDF {b_key} setting")
        for w_key in ("stem_weight", "skip_weight", "char_weight"):
            if w_key in config:
                wv = config[w_key]
                if type(wv) not in (int, float) or not math.isfinite(wv) or not 0.0 < float(wv) <= 5.0:
                    raise ValueError(f"Unsupported TF-IDF {w_key} setting")
        if "word_quota_frac" in config:
            wqf = config["word_quota_frac"]
            if type(wqf) not in (int, float) or not math.isfinite(wqf) or not 0.05 < float(wqf) < 0.98:
                raise ValueError("Unsupported TF-IDF word_quota_frac setting")
        terms = config["terms"]
        if not isinstance(terms, (list, tuple)) or not 0 < len(terms) <= limit:
            raise ValueError(f"TF-IDF vocabulary must contain 1–{limit} terms")
        return list(terms), limit

    @classmethod
    def from_config(cls, config: dict, *, idf_array: Any = None) -> TfidfProjector:
        terms, _ = cls.validate_header_config(config, allow_missing_idf=(idf_array is not None))
        if "idf" in config:
            idf_vals = config["idf"]
        elif idf_array is not None:
            arr = np.asarray(idf_array, dtype=np.float64)
            if arr.ndim != 1 or len(arr) != len(terms):
                raise ValueError("TF-IDF weights must match the vocabulary")
            idf_vals = tuple(float(v) for v in arr.tolist())
        else:
            raise ValueError("Unsupported TF-IDF settings")
        return cls(
            terms,
            idf_vals,
            char_ngrams=config.get("char_ngrams"),
            use_stems=config.get("use_stems", False),
            stem_weight=config.get("stem_weight", DEFAULT_STEM_WEIGHT),
            use_skip_bigrams=config.get("use_skip_bigrams", False),
            skip_weight=config.get("skip_weight", DEFAULT_SKIP_WEIGHT),
            word_quota_frac=config.get("word_quota_frac", DEFAULT_WORD_QUOTA_FRAC),
            char_weight=config.get("char_weight", DEFAULT_CHAR_WEIGHT),
        )

    from_texts = fit
    to_dict = to_config
    from_dict = from_config

