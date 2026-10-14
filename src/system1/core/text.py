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


def _terms(text: str, char_ngrams: tuple[int, int] | None = None) -> list[str]:
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
                score = num / den if den > 0.0 else 0.0
            else:  # "mi"
                score = 0.0
                for obs, row_sum, col_sum in (
                    (a, df, n_c),
                    (b, df, not_n_c),
                    (c_neg, not_df, n_c),
                    (d, not_df, not_n_c),
                ):
                    if obs > 0.0 and row_sum > 0.0 and col_sum > 0.0:
                        score += (obs / n_total) * math.log((n_total * obs) / (row_sum * col_sum))

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
        word_budget = min(len(word_terms), max(1, int(max_features * 0.62)))
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

    def __init__(
        self,
        terms: Sequence[str],
        idf: Sequence[float],
        *,
        char_ngrams: Sequence[int] | None = None,
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
        validated_ngrams = _validate_char_ngrams(char_ngrams)
        object.__setattr__(self, "_terms", tuple(terms))
        object.__setattr__(self, "_idf", tuple(float(value) for value in idf))
        object.__setattr__(self, "_char_ngrams", validated_ngrams)
        object.__setattr__(self, "_vocabulary", MappingProxyType({term: i for i, term in enumerate(terms)}))
        object.__setattr__(self, "_digest", hashlib.sha256(json.dumps(self.to_config(), sort_keys=True).encode()).hexdigest())

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
            terms = _terms(text, validated_ngrams)
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
            )
        return cls(
            selected,
            [math.log((1 + count) / (1 + document_frequency[term])) + 1 for term in selected],
            char_ngrams=validated_ngrams,
        )

    def raw_vector(self, text: str) -> np.ndarray:
        """Compute unnormalized TF-IDF vector for familiarity/support inspection."""
        vector = np.zeros(self.dimension, dtype=np.float32)
        term_counts = Counter(_terms(text, self._char_ngrams))
        negated_targets = {
            t[4:] for t in term_counts if t.startswith("neg:") and t in self._vocabulary
        }
        for term, count in term_counts.items():
            index = self._vocabulary.get(term)
            if index is not None:
                if term.startswith("#"):
                    scale = 0.32
                elif term.startswith("neg:") or term.startswith("not_"):
                    scale = 1.6
                elif term in negated_targets or any(w in negated_targets for w in term.split()):
                    scale = 0.15
                else:
                    scale = 1.0
                vector[index] = scale * (1 + math.log(count)) * self._idf[index]
        return vector

    def raw_norm(self, text: str) -> float:
        """Return pre-normalization L2 norm of matched TF-IDF features."""
        return float(np.linalg.norm(self.raw_vector(text)))

    def project(self, text: str, recency_weighted: bool | None = None) -> np.ndarray:
        if recency_weighted:
            raise ValueError("TF-IDF uses document frequencies, not recency weighting")
        vector = self.raw_vector(text)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def project_batch(self, texts: Sequence[str], recency_weighted: bool | None = None) -> np.ndarray:
        return np.stack([self.project(text, recency_weighted) for text in texts]) if len(texts) else np.empty((0, self.dimension), dtype=np.float32)

    def projector_digest(self) -> str:
        return self._digest

    def to_config(self) -> dict:
        cfg: dict[str, Any] = {"type": "tfidf", "terms": list(self._terms), "idf": list(self._idf)}
        if self._char_ngrams is not None:
            cfg["char_ngrams"] = list(self._char_ngrams)
        if len(self._terms) > DEFAULT_MAX_FEATURES:
            cfg["max_features"] = len(self._terms)
        return cfg

    @classmethod
    def from_config(cls, config: dict) -> TfidfProjector:
        if (
            not isinstance(config, dict)
            or not {"type", "terms", "idf"} <= set(config)
            or set(config) - {"type", "terms", "idf", "char_ngrams", "max_features"}
            or config["type"] != "tfidf"
        ):
            raise ValueError("Unsupported TF-IDF settings")
        limit = DEFAULT_MAX_FEATURES
        if "max_features" in config:
            mf = config["max_features"]
            if type(mf) is not int or not DEFAULT_MAX_FEATURES < mf <= MAX_FEATURES:
                raise ValueError("Unsupported TF-IDF max_features setting")
            limit = mf
        terms = config["terms"]
        if not isinstance(terms, (list, tuple)) or not 0 < len(terms) <= limit:
            raise ValueError(f"TF-IDF vocabulary must contain 1–{limit} terms")
        return cls(terms, config["idf"], char_ngrams=config.get("char_ngrams"))

    to_dict = to_config
    from_dict = from_config

