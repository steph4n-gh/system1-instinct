"""Local subword features and optional lexical/dense feature fusion.

The NumPy subword table is seeded locally and adjusted by curated semantic
anchors. It is not a downloaded or pretrained language model. MLX is optional;
no workload-independent latency bound is asserted."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import re
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    mx = None
    HAS_MLX = False


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
    "none",
    "nothing",
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

_ANCHOR_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
    "were", "have", "has", "had", "not", "never", "without", "all", "any",
    "our", "your", "their", "out", "off", "over", "under", "more", "less",
    "very", "too", "can", "will", "just", "been", "being", "onto", "upon",
    "a", "an", "in", "on", "at", "to", "of", "by", "as", "is", "be",
    "do", "does", "did", "would", "could", "should", "may", "might",
    "it", "its", "we", "you", "they", "them", "i", "me", "my", "or", "but", "if", "so",
})


_EMBEDDING_CACHE_MAXSIZE = 16384


def _stem_word_impl(word: str) -> str:
    """Deterministic Porter-style English morphological stemmer for cold-start alignment."""
    w = word.lower().strip()
    if len(w) <= 3 or not w.isalpha():
        return w
    # Step 1a: Plurals
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "i"
    elif w.endswith(("ches", "shes", "xes", "zes", "sses")) and len(w) > 5:
        w = w[:-2]
    elif w.endswith("ves") and len(w) > 4:
        w = w[:-3] + "f"
    elif w.endswith("s") and not w.endswith(("ss", "us", "is", "as", "os", "ys")) and len(w) > 3:
        w = w[:-1]

    # Step 1b: Nominalizations & verbal inflections
    if w.endswith(("ications", "ication")) and len(w) > 8:
        w = w[: -8 if w.endswith("s") else -7] + "i"
    elif w.endswith(("urations", "uration", "llations", "llation", "zations", "zation")) and len(w) > 8:
        w = w[: -6 if w.endswith("s") else -5]
    elif w.endswith(("ations", "ation", "itions", "ition", "utions", "ution", "etions", "etion")) and len(w) > 6:
        w = w[: -4 if w.endswith("s") else -3]
    elif w.endswith(("isons", "ison")) and len(w) > 6:
        w = w[: -5 if w.endswith("s") else -4]
    elif w.endswith(("ments", "ment")) and len(w) > 6:
        w = w[: -5 if w.endswith("s") else -4]
    elif w.endswith("ying") and len(w) > 5:
        w = w[:-4] + "i"
    elif w.endswith("ing") and len(w) >= 6:
        base = w[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        w = base
    elif w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "i"
    elif w.endswith("ed") and not w.endswith("eed") and len(w) >= 5:
        base = w[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        w = base

    # Step 1c & 5: Normalize trailing -y -> -i, silent -e, and double-l
    if len(w) > 4 and w.endswith("y"):
        w = w[:-1] + "i"
    if len(w) >= 4 and w.endswith("e") and not w.endswith(("ee", "oe", "ye")):
        w = w[:-1]
    if len(w) >= 4 and w.endswith("ll"):
        w = w[:-1]
    return w


_stem_word_cached = lru_cache(maxsize=_EMBEDDING_CACHE_MAXSIZE)(_stem_word_impl)


def _stem_word(word: str) -> str:
    """Deterministic Porter-style English morphological stemmer with bounded configurable cache."""
    return _stem_word_cached(word)


_stem_word.cache_clear = lambda: _stem_word_cached.cache_clear()  # type: ignore[attr-defined]
_stem_word.cache_info = lambda: _stem_word_cached.cache_info()  # type: ignore[attr-defined]


_CLAUSE_SPLIT_RE = re.compile(r"[.!?;,]+")


def _normalize_text(text: str) -> str:
    """Lowercase, normalize contractions, and bound input text length."""
    clean = text.lower().strip()
    if len(clean) > 8192:
        clean = clean[:4096] + " " + clean[-4096:]
    return _CONTRACTION_RE.sub(lambda m: m.group(0).replace("'", ""), clean)


def _find_negation_scope(words: Sequence[str]) -> Dict[int, str]:
    """Map token indices within 1-2 content tokens of a negation cue to that cue."""
    negated: Dict[int, str] = {}
    n_words = len(words)
    for i, w in enumerate(words):
        if w in _NEGATION_TOKENS:
            bound = 0
            for j in range(i + 1, min(i + 4, n_words)):
                target = words[j]
                if target in _NEGATION_TOKENS:
                    break
                if target in _NEGATION_SKIP_MODIFIERS:
                    continue
                negated[j] = w
                bound += 1
                if bound >= 2:
                    break
    return negated


def _tokenize_with_negation_scope(
    clean_text: str,
    pattern: str = r"\b\w+\b",
    max_len: Optional[int] = None,
) -> Tuple[List[str], Dict[int, str]]:
    """Tokenize text while confining negation scope to individual clauses."""
    words: List[str] = []
    negated: Dict[int, str] = {}
    for clause in _CLAUSE_SPLIT_RE.split(clean_text):
        if not clause:
            continue
        clause_words = re.findall(pattern, clause)
        if max_len is not None:
            clause_words = [w for w in clause_words if len(w) <= max_len]
        if not clause_words:
            continue
        offset = len(words)
        words.extend(clause_words)
        for idx, cue in _find_negation_scope(clause_words).items():
            negated[offset + idx] = cue
    return words, negated



_SEMANTIC_ANCHORS: Dict[str, List[str]] = {
    "rest_sleep_bedroom": [
        "sleep", "bed", "bedroom", "sack", "hay", "slumber", "rest", "nap", "tired",
        "night", "pillow", "mattress", "hitting the sack", "hit the hay", "turn in",
        "bedtime", "sleeping", "asleep", "shuteye", "doze", "drowsy", "bedside lamps",
        "master bedroom", "guest bedroom"
    ],
    "living_room": [
        "living room", "couch", "sofa", "television", "tv", "lounge", "media",
        "coffee table", "entertainment", "living area lights"
    ],
    "kitchen": [
        "kitchen", "cook", "cooking", "refrigerator", "fridge", "oven", "stove",
        "dishwasher", "microwave", "pantry", "dining", "meal", "coffee machine",
        "kitchen lights"
    ],
    "garage": [
        "garage", "driveway", "workshop", "overhead", "parking", "car", "vehicle",
        "exterior floodlights", "garage door"
    ],
    "whole_house": [
        "whole house", "entire residence", "all rooms", "every room", "all lights",
        "everywhere", "domicile", "house", "residence", "throughout the house",
        "all devices in the house"
    ],
    "device_lights": [
        "light", "lights", "lamp", "lamps", "bulb", "bulbs", "fixture", "fixtures",
        "led", "strip", "illumination", "sconce", "chandelier", "illuminate",
        "ceiling fixtures", "turn off lights", "turn on lights"
    ],
    "device_thermostat": [
        "thermostat", "temperature", "heating", "cooling", "hvac", "ac", "degrees",
        "climate control", "cooler", "heater", "furnace"
    ],
    "device_locks": [
        "lock", "locks", "locked", "unlock", "unlocked", "deadbolt", "bolt", "smart deadbolt",
        "front door lock", "secure door", "entryway"
    ],
    "device_blinds": [
        "blinds", "shades", "window shades", "roller blinds", "curtains", "drapery"
    ],
    "device_music": [
        "music", "audio", "speaker", "speakers", "volume", "sound", "spotify",
        "playlist", "track", "song"
    ],
    "action_turn_on": [
        "turn on", "power on", "activate", "illuminate", "open"
    ],
    "action_turn_off": [
        "turn off", "power off", "shut down", "extinguish", "deactivate", "lights out"
    ],
    "action_adjust_level": [
        "adjust level", "dim", "brighten", "raise temperature", "lower temperature", "change volume"
    ],
    "action_lock": [
        "lock door", "engage lock", "secure bolt", "deadbolt lock"
    ],
    "action_unlock": [
        "unlock door", "disengage lock", "unbolt door"
    ],
    "aml_normal": [
        "normal legitimate business", "payroll", "retail purchase", "recurring payment",
        "routine transfer", "standard invoice", "conforms to profile"
    ],
    "aml_structuring": [
        "structuring", "multiple sequential transactions", "below reporting threshold",
        "9850", "9900", "cash deposits", "smurf", "smurfing", "consecutive deposits",
        "branch deposits", "structuring suspect"
    ],
    "aml_sanctions": [
        "sanctions nexus", "ofac", "sanctioned entities", "sanctioned bank",
        "counterparties with sanctions", "wired funds offshore", "offshore wire",
        "embargo", "blacklist"
    ],
    "aml_velocity": [
        "unusual velocity", "rapid fund movement", "depositing and emptying",
        "rapid automated movement", "churning accounts", "high velocity transfers"
    ],
    "aml_high_risk_jurisdiction": [
        "high risk jurisdiction", "cross border transfers", "secrecy havens",
        "tax haven", "non cooperative jurisdiction", "cayman", "panama", "bvi"
    ],
    "aml_sar_filing": [
        "file sar", "suspicious activity report", "fincen", "mandatory sar filing",
        "illicit origin", "money laundering", "aml investigation", "freeze account"
    ],
    "code_security_pass": [
        "meets secure coding guidelines", "clean code diff", "parameterized query",
        "prepared statement", "safe merge", "pass security"
    ],
    "code_security_sql_injection": [
        "sql injection", "sqli", "unsanitized user string", "concat inside database query",
        "select * from", "execute query", "or 1=1", "f\"select * from"
    ],
    "code_security_secret": [
        "hardcoded secret", "plaintext secret", "api key in code", "private key plaintext",
        "password token", "credentials in source"
    ],
    "code_security_traversal": [
        "path traversal", "arbitrary file read", "relative directory paths", "../",
        "dot dot slash", "lfi traversal"
    ],
    "code_security_deserialization": [
        "unsafe deserialization", "pickle load", "yaml load untrusted", "untrusted payload",
        "remote code execution deserialization", "rce deserialization"
    ],
    "code_security_critical_block": [
        "critical block", "block ci merge", "cvss 9", "critical vulnerability", "block merge"
    ],
    "rag_direct_answer": [
        "direct answer", "exact facts", "answering query", "hours of continuous active playback",
        "battery life", "complete factual ground truth", "exact figures"
    ],
    "rag_contextually_related": [
        "contextually related", "general subject area", "lacks specific answer",
        "discusses topic generally", "needs more context"
    ],
    "rag_irrelevant": [
        "irrelevant", "no bearing on user query", "unrelated passage", "off topic noise"
    ],
    "insurance_straight_through": [
        "straight through approval", "instant payout", "low value standard claim",
        "cracked windshield", "stray pebble", "clean history", "repair quote $320",
        "no injuries"
    ],
    "insurance_desk_adjuster": [
        "desk adjuster review", "moderate damage", "repair quote validation", "manual policy review"
    ],
    "insurance_special_investigations": [
        "special investigations unit", "siu", "fraud risk", "staged accident",
        "falsified invoice", "potential fraud true", "suspicious claim"
    ],
    "insurance_catastrophe": [
        "urgent catastrophe", "total structural destruction", "bodily injury",
        "emergency hospital", "catastrophic damage"
    ],
    "support_technical": [
        "technical support", "software bug", "application error", "crash dump",
        "nullpointerexception", "stack trace", "api error 500", "server outage",
        "database connection failure", "unhandled exception", "system crash",
        "service unavailable", "broken feature", "defect", "malfunction"
    ],
    "support_billing": [
        "billing inquiry", "invoice dispute", "credit card charge", "subscription fee",
        "refund request", "unauthorized charge", "chargeback", "payment receipt",
        "cancel subscription", "billing account", "double charged", "overcharged",
        "billed twice", "monthly invoice", "billing statement"
    ],
    "support_account_auth": [
        "password reset", "mfa token", "locked out of account", "login failed",
        "two factor authentication", "change email address", "forgot password",
        "account recovery", "profile update", "cannot login", "cant sign in"
    ],
    "firewall_allow_benign": [
        "allow immediate", "benign routine", "read only", "readme", "inspect repository",
        "pytest", "unit tests", "git log", "git status", "lsof", "ps aux", "black --check",
        "ast parse", "ast.parse", "clean code diff", "syntax check", "status query",
        "harmless routine", "low risk read only"
    ],
    "firewall_require_human": [
        "require human approval", "privileged admin", "elevated privilege", "destructive modification",
        "restart container", "docker restart", "schema migration", "alembic upgrade", "modify config",
        "update .env", "elasticsearch query", "tls certificate", "openssl req", "pip install",
        "rotate credentials", "git push", "push remote"
    ],
    "firewall_quarantine": [
        "quarantine tool execution", "anomalous tool invocation", "suspicious pipeline",
        "pipe execution", "curl | bash", "binary dropper", "chmod +x /dev/shm", "in-memory payload",
        "blind sqli", "pg_sleep", "graphql introspection", "discover hidden endpoints",
        "s3 unapproved bucket", "external untrusted bucket"
    ],
    "firewall_terminate_exploit": [
        "terminate agent session", "critical exploit", "credential harvesting", "prompt injection",
        "reverse shell", "data exfiltration", "cat /etc/shadow", "private key theft", "jailbreak",
        "steal ssh key", "cat ~/.ssh/id_rsa", "dump memory", "gcore", "sql injection", "drop table",
        "pii leak", "ssn credit card", "hipaa breach", "biometric exfil"
    ],
    "threat_data_exfiltration": [
        "data exfiltration", "covert channel", "outbound curl", "upload s3", "dns tunnel",
        "dig +short c2", "scp private key", "data drop", "payload dump", "transmit credentials",
        "nc 203.0.113", "exfiltrate", "outbound payload", "c2-collector"
    ],
    "threat_unauthorized_shell": [
        "unauthorized shell execution", "spawn bash", "interactive shell", "nc -e /bin/bash",
        "reverse shell", "subshell execution", "bash -i", "sh -c", "eval python", "os.system",
        "subprocess.popen", "base64 -d | sh", "ld_preload", "crontab persistence", "visudo backdoor"
    ],
    "threat_credential_access": [
        "credential access", "steal ssh key", "cat ~/.ssh/id_rsa", "cat ~/.aws/credentials",
        "grep sk_live", "k8s service account token", "dump memory", "gcore", "api key theft",
        "extract passwords", "private_key", "jwt_token"
    ],
    "threat_pii_leak": [
        "pii leak", "customer profiles", "ssn", "tax_id", "credit card number", "medical records",
        "hipaa breach", "patient telemetry", "biometrics", "payroll export", "salary leak",
        "personally identifiable", "hr_payroll_vault", "oncology diagnosis"
    ],
    "threat_sql_injection": [
        "sql injection", "sqli", "' or 1=1", "drop table", "pg_sleep", "union select",
        "blind sql injection", "auth bypass sql", "update accounts set balance", "drop table audit"
    ],
    "threat_prompt_injection": [
        "prompt injection", "jailbreak", "ignore previous instructions", "system prompt exfiltration",
        "override safety guardrails", "developer mode activated", "dan mode", "do anything now",
        "disregard prior instructions", "unfiltered master system prompt", "bypass safety filters",
        "reveal hidden system instructions"
    ],
    "billing_refunds": [
        "refund", "reimbursement", "money back", "return policy", "credit back",
        "full refund", "partial refund", "reimburse", "rebate", "issue credit",
        "refund status", "reversed charge", "return item", "cancel for refund",
        "requesting refund", "refund my payment"
    ],
    "account_lifecycle": [
        "account signup", "create account", "onboarding", "register profile",
        "close account", "delete account", "deactivate account", "suspend account",
        "reactivate account", "account termination", "upgrade plan", "downgrade plan",
        "transfer ownership", "account settings", "user provisioning"
    ],
    "shipping_logistics": [
        "shipping", "delivery", "package", "parcel", "courier", "tracking number",
        "shipment delayed", "lost in transit", "warehouse dispatch", "customs clearance",
        "out for delivery", "carrier status", "doorstep delivery", "freight",
        "backorder", "fulfillment", "missing package", "damaged in transit",
        "estimated delivery", "shipping address", "never arrived", "not delivered"
    ],
    "sentiment_positive": [
        "helpful", "very helpful", "satisfied", "great service", "working perfectly",
        "resolved quickly", "smooth experience", "pleased", "appreciate", "thank you",
        "awesome", "reliable", "fixed", "happy", "excellent", "wonderful", "delighted",
        "works great", "impressed", "love it"
    ],
    "sentiment_negative": [
        "frustrated", "angry", "disappointed", "terrible service", "unacceptable",
        "useless", "broken", "worst experience", "not working", "never arrived",
        "not helpful", "isnt working", "wasnt helpful", "didnt arrive", "doesnt work",
        "wont work", "cant use", "without help", "unhelpful", "dissatisfied",
        "furious", "horrible", "awful", "fails constantly", "waste of time"
    ],
    "urgency_high": [
        "urgent", "emergency", "critical", "immediately", "asap", "right now",
        "production down", "blocker", "sev1", "p0", "time sensitive", "high priority",
        "escalate immediately", "deadline today", "severe outage", "stop everything"
    ],
    "urgency_low": [
        "whenever possible", "low priority", "no rush", "minor cosmetic",
        "feature request", "backlog", "nice to have", "non urgent", "informational",
        "routine question", "at your convenience", "someday", "trivial"
    ],
    "banking_finance": [
        "wire transfer", "ach transfer", "bank account", "routing number",
        "ledger balance", "overdraft fee", "interest rate", "card declined",
        "debit card", "credit limit", "statement balance", "deposit check",
        "withdrawal", "foreign exchange", "swift code", "iban", "loan payment",
        "mortgage", "treasury", "checking account", "savings account"
    ],
    "operations_devops": [
        "deployment", "rollback", "incident response", "sla breach", "capacity planning",
        "autoscaling", "maintenance window", "compliance check",
        "workflow automation", "provisioning", "kubernetes cluster", "ci cd pipeline",
        "monitoring alert", "runbook", "oncall", "infrastructure", "load balancer"
    ],
    "route_mid_standard": [
        "standard model", "mid tier model", "balanced model",
        "write function", "refactor code", "unit test generation", "api endpoint",
        "sql query writing", "bug fix", "code review", "pull request review",
        "draft documentation", "moderate complexity", "everyday coding"
    ],
}

_COLD_START_CONCEPT_CLUSTERS: Dict[str, List[str]] = {
    "route_fast_simple": [
        "haiku", "greeting", "spellcheck", "lookup", "tldr",
        "quick lookup", "simple question", "factual answer", "fast model",
        "small model", "low latency response", "lightweight task", "summarize text",
        "translate sentence", "extract entities", "format json",
        "yes or no", "true or false", "one sentence summary", "define term"
    ],
    "route_frontier_reasoning": [
        "opus", "theorem", "byzantine", "paxos", "cryptography",
        "complex mathematical proof", "multi step reasoning", "frontier reasoning model",
        "theorem proving", "hard logic puzzle", "advanced algorithmic synthesis",
        "deep analysis", "cot reasoning", "formal verification",
        "byzantine fault tolerance", "distributed consensus", "lock free concurrency",
        "cryptographic protocol", "security architecture audit", "compiler optimization"
    ],
    "route_mid_standard": [
        "sonnet", "refactor", "endpoint", "docstring",
        "standard model", "mid tier model", "balanced model",
        "write function", "refactor code", "unit test generation", "api endpoint",
        "sql query writing", "bug fix", "code review", "pull request review",
        "draft documentation", "moderate complexity", "everyday coding",
        "implement class", "parse csv", "react component", "fastapi route"
    ],
    "support_tech_extended": [
        "bug", "crash", "error", "exception", "traceback", "timeout", "latency",
        "webhook", "sdk", "api", "oauth", "sso", "saml", "mfa", "cors", "ssl", "tls",
        "websocket", "graphql", "login", "authentication", "password", "documentation",
        "slow", "freezing", "glitch", "malfunction", "defect", "debugging",
        "system bug", "api error", "502 error", "500 error", "503 error", "504 error",
        "400 error", "401 error", "403 error", "404 error", "429 error",
        "integration issue", "sdk question", "webhook problem", "webhook delivery",
        "performance problem", "slow loading", "high latency",
        "feature question", "configuration help", "how to configure",
        "login issue", "access problem", "password reset",
        "two factor", "locked out", "stack trace", "memory leak", "cpu spike",
        "connection refused", "ssl certificate", "tls handshake", "cors error",
        "rate limit", "rest api", "cli tool", "sync error", "export failure",
        "import error", "dashboard loading", "broken link", "ui glitch"
    ],
    "support_billing_extended": [
        "invoice", "billing", "billed", "refund", "reimburse", "reimbursement",
        "overcharged", "chargeback", "prorated", "receipt", "autopay", "overdue",
        "subscription", "vat", "gst", "tax", "statement", "credit", "debit", "payment",
        "invoice dispute", "duplicate charge", "refund request", "payment method",
        "billing address", "pricing question", "plan comparison", "subscription cancellation",
        "prorated refund", "payment failure", "credit card", "debit card",
        "double charged", "billed twice", "monthly invoice", "annual invoice",
        "payment receipt", "card declined", "expired card", "tax exemption",
        "vat number", "purchase order", "w9 form", "billing statement",
        "overdue balance", "credit memo", "downgrade plan", "billing cycle",
        "unexpected charge", "unauthorized charge", "cancel subscription",
        "wire payment", "ach billing"
    ],
    "sales_commercial": [
        "enterprise", "licensing", "licenses", "seats", "demo", "competitor",
        "procurement", "rfp", "rfq", "quote", "proposal", "reseller", "partner",
        "onboarding", "pilot", "poc", "soc2", "hipaa", "baa", "iso27001", "fedramp",
        "discount", "expansion", "upgrade", "multi-year", "non-profit",
        "enterprise seat", "volume licensing", "product demo", "competitor comparison",
        "custom pricing", "non profit discount", "academic discount", "multi year contract",
        "partner program", "reseller program", "upgrade inquiry", "plan expansion",
        "bulk discount", "commercial contract", "enterprise quote", "sales inquiry",
        "request for proposal", "annual commitment", "trial extension", "proof of concept",
        "additional seats", "team licenses", "enterprise plan", "enterprise tier",
        "dedicated instance", "on premise deployment", "self hosted", "custom sla",
        "security questionnaire", "soc2 report", "hipaa baa", "vendor onboarding",
        "evaluating your platform", "migrate from competitor", "schedule a demo",
        "volume discount", "enterprise contract"
    ],
    "support_escalate": [
        "escalate", "escalation", "manager", "supervisor", "executive", "lawyer",
        "attorney", "lawsuit", "litigation", "unacceptable", "furious", "outraged",
        "incompetent", "sla", "breach", "regulatory", "ftc", "bbb", "ciso",
        "demand manager", "speak to supervisor", "talk to a manager", "escalate immediately",
        "repeated unresolved", "formal complaint", "sla violation", "sla breach",
        "legal threat", "legal action", "regulatory complaint", "gdpr complaint",
        "attorney general", "data breach", "security incident", "compromised account",
        "unauthorized access", "executive engagement", "vp of", "chief executive",
        "high financial impact", "revenue loss", "costing us", "losing money",
        "third time", "fourth time", "fifth time", "multiple tickets", "still not resolved",
        "unresolved for days", "unresolved for weeks", "no response", "ignored for",
        "breach of contract", "cancel our enterprise", "pulling our account", "going public"
    ],
    "ops_readonly_inspect": [
        "view", "viewing", "list", "listing", "read", "reading", "show", "display",
        "displaying", "inspect", "inspecting", "lint", "linting", "pytest", "unittest",
        "jest", "mypy", "pyright", "flake8", "eslint", "ruff", "pylint", "ripgrep",
        "grep", "diff", "uptime", "whoami", "uname", "hostname", "lsof", "netstat",
        "traceroute", "nslookup", "diagnostic", "readonly",
        "safe read only", "diagnostic operation", "viewing files", "listing directories",
        "reading logs", "running tests", "type checking",
        "displaying system information", "git status", "git diff", "git log",
        "git branch", "git show", "git blame", "unit tests", "test suite",
        "cargo test", "go test", "npm test", "black check", "tsc noemit",
        "prettier check", "inspect file", "list files", "read file", "view file",
        "show logs", "check status", "disk usage", "memory usage", "process list",
        "search text", "grep pattern", "find files", "count lines", "syntax check",
        "health check", "ping host", "describe resource", "kubectl get",
        "kubectl describe", "kubectl logs", "docker ps", "docker logs",
        "docker images", "docker inspect", "ps aux", "cat file", "head lines", "tail logs"
    ],
    "ops_modify_configure": [
        "modify", "modifying", "configure", "configuring", "configuration",
        "install", "installing", "create", "creating", "edit", "editing",
        "update", "updating", "write", "writing", "patch", "patching",
        "mkdir", "touch", "sed", "tee", "uvicorn", "gunicorn", "alembic",
        "prisma", "migration", "migrate", "webhook", "crontab", "dockerfile",
        "needs human review", "modifying configuration", "installing packages",
        "creating files", "modifying files", "network requests", "internal endpoints",
        "running development servers", "docker containers", "database writes",
        "development environments", "pip install", "npm install", "apt install",
        "apt get install", "brew install", "yarn add", "pnpm add", "cargo add",
        "go get", "gem install", "composer require", "poetry add", "conda install",
        "update config", "edit file", "write file", "append file", "patch file",
        "create directory", "copy file", "move file", "rename file",
        "chmod +x", "chmod 644", "chmod 600",
        "start server", "restart service", "stop service", "systemctl restart",
        "systemctl start", "systemctl stop", "service restart",
        "docker run", "docker build", "docker start", "docker stop", "docker restart",
        "docker compose", "kubectl apply", "kubectl scale", "kubectl rollout",
        "flask run", "npm run dev", "npm start", "schema migration",
        "alembic upgrade", "prisma migrate", "database migration",
        "insert into", "update set", "create table", "alter table", "create index",
        "curl post", "curl put", "post request", "put request", "internal api",
        "environment variable", "update env", "nginx conf", "tsconfig",
        "git commit", "git push", "git checkout", "git merge", "git rebase",
        "cron job", "port forwarding", "staging environment"
    ],
    "ops_destructive_exploit": [
        "destructive", "malicious", "dangerous", "delete", "deleting", "wipe", "wiping",
        "exfiltrate", "exfiltrating", "exfiltration", "exploit", "exploits",
        "malware", "rootkit", "rootkits", "ransomware", "jailbreak", "backdoor",
        "trojan", "keylogger", "shellcode", "metasploit", "msfvenom", "sqli",
        "sqlmap", "xmrig", "minerd", "cryptominer", "shred", "mkfs", "fdisk",
        "sudoers", "nopasswd", "visudo", "id_rsa", "shadow", "nmap", "masscan",
        "hydra", "medusa", "hashcat", "netcat", "ncat", "socat", "ptrace", "insmod",
        "gcore", "chroot", "nsenter", "suid", "setuid",
        "destructive operation", "malicious operation", "dangerous operation",
        "deleting system files", "wiping disks", "rm -rf", "rm -fr",
        "exfiltrating secrets", "ssh keys", "api keys", "credentials", "private key",
        "reverse shells", "bind shell", "prompt injection", "jailbreak attempts",
        "ignore previous instructions", "ignore all prior", "disregard previous",
        "system prompt", "developer mode", "dan mode", "bypass safety",
        "override safety", "override instructions", "forget your rules",
        "reveal instructions", "unrestricted mode",
        "unauthorized privilege escalation", "chmod 777", "chmod +s", "chmod 4755",
        "chmod u+s", "chown root", "sudo su", "sudo -i", "passwd root",
        "privileged container", "docker sock", "cap_setuid",
        "data destruction", "encrypt files", "openssl enc", "bitcoin", "monero",
        "ransom note", "stratum", "sql injection", "unauthorized database manipulation",
        "drop table", "drop database", "truncate table", "delete from", "union select",
        "or 1=1", "xp_cmdshell", "into outfile", "load_file", "pg_sleep",
        "etc shadow", "etc passwd", "etc sudoers", "id_ed25519", "authorized_keys",
        "aws credentials", "kube config", "npmrc", "dump memory",
        "meta data", "webhook site", "pastebin", "ngrok", "dns tunnel",
        "nc -e", "bash -i", "dev tcp", "curl bash", "curl sh", "wget bash", "wget sh",
        "base64 -d", "ld_preload", "modprobe", "kernel module",
        "arp spoof", "syn flood", "ddos", "hping3", "clear history", "history -c",
        "unset histfile", "disable firewall", "ufw disable", "iptables -f",
        "setenforce 0", "disable antivirus", "disable defender", "format disk",
        "dd if", "fork bomb", "shutdown -h", "reboot -f", "kill -9"
    ],
}

_ALL_CONCEPT_CLUSTERS: Dict[str, List[str]] = {
    **_SEMANTIC_ANCHORS,
    **_COLD_START_CONCEPT_CLUSTERS,
}

_EXTENDED_V1_4_CLUSTERS: frozenset[str] = frozenset({
    "billing_refunds",
    "account_lifecycle",
    "shipping_logistics",
    "sentiment_positive",
    "sentiment_negative",
    "urgency_high",
    "urgency_low",
    "banking_finance",
    "operations_devops",
    "route_mid_standard",
})

_ANTIPODAL_ANCHOR_PAIRS: Dict[str, str] = {
    "sentiment_negative": "sentiment_positive",
    "urgency_low": "urgency_high",
}

_NEGATED_CONCEPT_TARGETS: Dict[str, List[str]] = {
    "sentiment_negative": [
        "working", "work", "works", "worked", "helpful", "help", "arrived",
        "arrive", "delivered", "deliver", "received", "receive", "resolved",
        "resolve", "fixed", "fix", "satisfied", "happy", "good", "great",
        "useful", "reliable", "responding", "respond", "loading", "load",
    ],
    "support_account_auth": [
        "login", "log", "signin", "sign", "access", "authenticate", "unlock",
    ],
}

_CONCEPT_FAMILIES: Tuple[Tuple[str, ...], ...] = (
    ("ops_readonly_inspect", "ops_modify_configure", "ops_destructive_exploit"),
    ("support_tech_extended", "support_billing_extended", "sales_commercial", "support_escalate"),
    ("route_fast_simple", "route_mid_standard", "route_frontier_reasoning"),
    ("sentiment_positive", "sentiment_negative"),
    ("urgency_high", "urgency_low"),
    ("rest_sleep_bedroom", "living_room", "kitchen", "garage", "whole_house"),
    ("device_lights", "device_thermostat", "device_locks", "device_blinds", "device_music"),
    ("aml_normal", "aml_structuring", "aml_sanctions", "aml_velocity", "aml_high_risk_jurisdiction", "aml_sar_filing"),
)

_COUPLED_CLUSTER_PARENTS: Dict[str, str] = {
    "billing_refunds": "support_billing",
    "support_billing_extended": "support_billing",
    "support_tech_extended": "support_technical",
    "ops_readonly_inspect": "firewall_allow_benign",
    "ops_modify_configure": "firewall_require_human",
    "ops_destructive_exploit": "firewall_terminate_exploit",
}


def _build_concept_stem_index() -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[Tuple[str, str], List[str]]]:
    """Pre-index semantic anchor clusters within orthogonal decision families.

    Single-word entries act as primary stem triggers (weight 1.35), while words
    occurring only inside multi-word phrases receive weak contextual weight (0.35)
    and primary pair weight (2.2) when their phrase pair co-occurs.
    """
    stem_weights: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    pair_clusters: Dict[Tuple[str, str], set[str]] = defaultdict(set)

    for family in _CONCEPT_FAMILIES:
        primary_stems: Dict[str, set[str]] = defaultdict(set)
        phrase_stems: Dict[str, set[str]] = defaultdict(set)
        for cluster_name in family:
            terms = _ALL_CONCEPT_CLUSTERS.get(cluster_name, ())
            for term in terms:
                words = [
                    w for w in re.findall(r"\b\w+\b", term.lower())
                    if w not in _ANCHOR_STOPWORDS and len(w) >= 2
                ]
                stems = [_stem_word(w) for w in words]
                if len(stems) == 1 and len(stems[0]) >= 2:
                    primary_stems[stems[0]].add(cluster_name)
                elif len(stems) >= 2:
                    for i in range(len(stems) - 1):
                        pair_clusters[(stems[i], stems[i + 1])].add(cluster_name)
                    for s in stems:
                        if len(s) >= 3:
                            phrase_stems[s].add(cluster_name)

        for s, cset in primary_stems.items():
            if len(cset) == 1:
                for c in sorted(cset):
                    stem_weights[s].append((c, 1.35))
            elif len(cset) == 2 and len(family) >= 3:
                for c in sorted(cset):
                    stem_weights[s].append((c, 0.55))

        for s, cset in phrase_stems.items():
            if s in primary_stems:
                continue
            if len(cset) == 1:
                for c in sorted(cset):
                    stem_weights[s].append((c, 0.35))

    pair_map: Dict[Tuple[str, str], List[str]] = {
        pair: sorted(cset) for pair, cset in pair_clusters.items()
    }
    return dict(stem_weights), pair_map


def _build_multiword_concept_index() -> Dict[str, Tuple[Tuple[str, str], ...]]:
    by_space3: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for cluster_name, terms in _ALL_CONCEPT_CLUSTERS.items():
        for term in terms:
            if " " in term:
                s = term.find(" ")
                if 0 < s < len(term) - 1:
                    by_space3[term[s - 1 : s + 2]].append((cluster_name, term))
    return {k: tuple(v) for k, v in by_space3.items()}


_CONCEPT_STEM_INDEX, _CONCEPT_PAIR_INDEX = _build_concept_stem_index()
_MULTIWORD_CONCEPTS_BY_SPACE3 = _build_multiword_concept_index()


def match_semantic_concepts(text: str) -> Dict[str, float]:
    """Match input text against general semantic concept clusters with lattice risk dominance."""
    if not text or not text.strip():
        return {}
    clean_text = _normalize_text(text)
    words, negated = _tokenize_with_negation_scope(clean_text)
    if not words:
        return {}

    scores: Dict[str, float] = defaultdict(float)

    # 1. Multi-word phrase matches via space-adjacent 3-gram index
    if " " in clean_text:
        seen_k3: set[str] = set()
        pos_sp = clean_text.find(" ")
        while pos_sp != -1:
            if 0 < pos_sp < len(clean_text) - 1:
                k3 = clean_text[pos_sp - 1 : pos_sp + 2]
                if k3 not in seen_k3:
                    seen_k3.add(k3)
                    cands = _MULTIWORD_CONCEPTS_BY_SPACE3.get(k3)
                    if cands is not None:
                        for cluster_name, term in cands:
                            if term in clean_text:
                                scores[cluster_name] += 2.6
            pos_sp = clean_text.find(" ", pos_sp + 1)

    # 2. Content stems & adjacent stem pairs (excluding negated positions)
    active_stems: List[str] = []
    for pos, w in enumerate(words):
        if pos in negated:
            target_st = _stem_word(w)
            for t_cluster, pos_words in _NEGATED_CONCEPT_TARGETS.items():
                if w in pos_words or target_st in {_stem_word(pw) for pw in pos_words}:
                    scores[t_cluster] += 2.2
            continue
        if w in _ANCHOR_STOPWORDS or len(w) < 2:
            continue
        st = _stem_word(w)
        active_stems.append(st)
        for c_name, w_val in _CONCEPT_STEM_INDEX.get(st, ()):
            scores[c_name] += w_val

    for i in range(len(active_stems) - 1):
        pair = (active_stems[i], active_stems[i + 1])
        for c_name in _CONCEPT_PAIR_INDEX.get(pair, ()):
            scores[c_name] += 2.2
        if i + 2 < len(active_stems):
            skip_pair = (active_stems[i], active_stems[i + 2])
            for c_name in _CONCEPT_PAIR_INDEX.get(skip_pair, ()):
                scores[c_name] += 1.4

    # 3. Lattice risk & escalation dominance:
    # Security hierarchy: Destructive/Exploit > Modify/Configure > Read-Only/Inspect
    destructive_score = (
        scores.get("ops_destructive_exploit", 0.0)
        + scores.get("firewall_terminate_exploit", 0.0)
        + scores.get("threat_data_exfiltration", 0.0)
        + scores.get("threat_unauthorized_shell", 0.0)
        + scores.get("threat_credential_access", 0.0)
        + scores.get("threat_sql_injection", 0.0)
        + scores.get("threat_prompt_injection", 0.0)
    )
    modify_score = (
        scores.get("ops_modify_configure", 0.0)
        + scores.get("firewall_require_human", 0.0)
    )
    if destructive_score >= 1.30:
        scores["ops_destructive_exploit"] = max(scores.get("ops_destructive_exploit", 0.0), destructive_score)
        for low_c in ("ops_readonly_inspect", "firewall_allow_benign", "ops_modify_configure", "firewall_require_human", "operations_devops"):
            scores.pop(low_c, None)
    elif modify_score >= 1.30:
        scores["ops_modify_configure"] = max(scores.get("ops_modify_configure", 0.0), modify_score)
        for low_c in ("ops_readonly_inspect", "firewall_allow_benign"):
            scores.pop(low_c, None)

    # Support routing hierarchy: Executive/Legal/Crisis Escalation overrides routine queues when strong
    esc_score = scores.get("support_escalate", 0.0)
    if esc_score >= 2.0:
        for routine_c in ("support_tech_extended", "support_technical", "support_billing_extended", "support_billing", "billing_refunds", "sales_commercial"):
            if scores.get(routine_c, 0.0) < esc_score * 1.15:
                scores.pop(routine_c, None)

    # Propagate coupled child clusters to canonical parent clusters
    for child_c, parent_c in _COUPLED_CLUSTER_PARENTS.items():
        if child_c in scores:
            scores[parent_c] = max(scores.get(parent_c, 0.0), scores[child_c] * 0.9)

    return dict(scores)


def _cached_token_hash_impl(token_norm: str, vocab_size: int) -> int:
    h = hashlib.sha256(token_norm.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % vocab_size


_cached_token_hash_cached = lru_cache(maxsize=_EMBEDDING_CACHE_MAXSIZE)(_cached_token_hash_impl)


def _cached_token_hash(token_norm: str, vocab_size: int) -> int:
    return _cached_token_hash_cached(token_norm, vocab_size)


_cached_token_hash.cache_clear = lambda: _cached_token_hash_cached.cache_clear()  # type: ignore[attr-defined]
_cached_token_hash.cache_info = lambda: _cached_token_hash_cached.cache_info()  # type: ignore[attr-defined]


def _sparse_word_hashes_impl(token: str, sparse_dim: int, count: int = 4) -> Tuple[Tuple[int, float], ...]:
    h = hashlib.sha256(f"word:{token}".encode("utf-8")).digest()
    res: List[Tuple[int, float]] = []
    for i in range(count):
        chunk = h[i * 4 : (i + 1) * 4]
        val = int.from_bytes(chunk, byteorder="big", signed=False)
        idx = val % sparse_dim
        sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
        res.append((idx, sign))
    return tuple(res)


def _sparse_single_hash_impl(prefix_token: str, sparse_dim: int) -> Tuple[int, float]:
    h = hashlib.sha256(prefix_token.encode("utf-8")).digest()
    idx = int.from_bytes(h[:4], "big") % sparse_dim
    sign = 1.0 if (h[4] & 1) == 0 else -1.0
    return idx, sign


_sparse_word_hashes_cached = lru_cache(maxsize=_EMBEDDING_CACHE_MAXSIZE)(_sparse_word_hashes_impl)
_sparse_single_hash_cached = lru_cache(maxsize=_EMBEDDING_CACHE_MAXSIZE)(_sparse_single_hash_impl)


def set_embedding_cache_maxsize(maxsize: int) -> None:
    """Update maxsize for module-level token/stem/subword LRU caches and flush old entries."""
    global _EMBEDDING_CACHE_MAXSIZE
    global _stem_word_cached, _cached_token_hash_cached
    global _sparse_word_hashes_cached, _sparse_single_hash_cached
    if type(maxsize) is not int or maxsize < 0:
        raise ValueError(f"maxsize must be a non-negative integer, got {maxsize}")
    _EMBEDDING_CACHE_MAXSIZE = int(maxsize)
    _stem_word_cached = lru_cache(maxsize=_EMBEDDING_CACHE_MAXSIZE)(_stem_word_impl)
    _cached_token_hash_cached = lru_cache(maxsize=_EMBEDDING_CACHE_MAXSIZE)(_cached_token_hash_impl)
    _sparse_word_hashes_cached = lru_cache(maxsize=_EMBEDDING_CACHE_MAXSIZE)(_sparse_word_hashes_impl)
    _sparse_single_hash_cached = lru_cache(maxsize=_EMBEDDING_CACHE_MAXSIZE)(_sparse_single_hash_impl)


def clear_embedding_caches() -> None:
    """Clear all module-level token/stem/subword LRU caches."""
    _stem_word_cached.cache_clear()
    _cached_token_hash_cached.cache_clear()
    _sparse_word_hashes_cached.cache_clear()
    _sparse_single_hash_cached.cache_clear()


class SubwordSemanticEmbeddings:
    """Seeded NumPy subword table with curated semantic anchors and negation-scope binding.

    The default table dimension is 256. Its construction does not teach a
    general language model; task quality requires separate evaluation."""

    def __init__(
        self,
        dimension: int = 256,
        vocab_size: int = 25000,
        seed: int = 42,
    ) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        self.dimension = dimension
        self.vocab_size = vocab_size
        self.seed = seed
        self._multiword_by_space3: Dict[str, Tuple[Tuple[str, bool, int], ...]] = self._build_multiword_anchor_index()
        self._table: np.ndarray = self._init_embedding_table()

    def _build_multiword_anchor_index(self) -> Dict[str, Tuple[Tuple[str, bool, int], ...]]:
        by_space3: Dict[str, List[Tuple[str, bool, int]]] = defaultdict(list)
        for terms in _SEMANTIC_ANCHORS.values():
            for term in terms:
                if " " in term:
                    s = term.find(" ")
                    if 0 < s < len(term) - 1:
                        tw = term.split()
                        first_is_neg = bool(tw and tw[0] in _NEGATION_TOKENS)
                        by_space3[term[s - 1 : s + 2]].append((term, first_is_neg, self._hash_token(term)))
        return {k: tuple(v) for k, v in by_space3.items()}

    @property
    def concept_clusters(self) -> Mapping[str, List[str]]:
        return _SEMANTIC_ANCHORS

    def _hash_token(self, token: str) -> int:
        """Deterministically map a token or n-gram string to a table row index."""
        return _cached_token_hash(token.strip().lower(), self.vocab_size)

    def _build_default_table(self) -> np.ndarray:
        """Constructs the dense subword embedding table with calibrated semantic anchors."""
        rng = np.random.RandomState(self.seed)
        # Initialize pseudo-random Gaussian base table
        table = rng.normal(0.0, 1.0, size=(self.vocab_size, self.dimension)).astype(np.float32)
        row_norms = np.linalg.norm(table, axis=1, keepdims=True)
        row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
        table /= row_norms

        anchor_centroids: Dict[str, np.ndarray] = {}
        protected_word_indices: set[int] = set()

        # Blend semantic anchor concept centroids
        for anchor_name, terms in _SEMANTIC_ANCHORS.items():
            is_extended = anchor_name in _EXTENDED_V1_4_CLUSTERS
            opposite_name = _ANTIPODAL_ANCHOR_PAIRS.get(anchor_name) if is_extended else None
            if opposite_name is not None and opposite_name in anchor_centroids:
                noise = rng.normal(0.0, 1.0, size=(self.dimension,)).astype(np.float32)
                n_norm = float(np.linalg.norm(noise))
                if n_norm > 1e-12:
                    noise /= n_norm
                c_vec = -0.90 * anchor_centroids[opposite_name] + 0.10 * noise
            else:
                c_vec = rng.normal(0.0, 1.0, size=(self.dimension,)).astype(np.float32)
            c_norm = float(np.linalg.norm(c_vec))
            if c_norm > 1e-12:
                c_vec /= c_norm
            anchor_centroids[anchor_name] = c_vec

            if anchor_name == "billing_refunds" and "support_billing" in anchor_centroids:
                c_vec = 0.65 * anchor_centroids["support_billing"] + 0.35 * c_vec
                c_norm = float(np.linalg.norm(c_vec))
                if c_norm > 1e-12:
                    c_vec /= c_norm
                anchor_centroids[anchor_name] = c_vec

            for term in terms:
                clean_term = _normalize_text(term) if is_extended else term
                idx = self._hash_token(clean_term)
                if not is_extended or idx not in protected_word_indices:
                    blended = 0.15 * table[idx] + 0.85 * c_vec
                    b_norm = float(np.linalg.norm(blended))
                    table[idx] = blended / (b_norm if b_norm > 1e-12 else 1.0)
                    if not is_extended:
                        protected_word_indices.add(idx)

                # Also blend individual subwords and word tokens of multi-word terms
                words = re.findall(r"\b\w+\b", clean_term.lower())
                if len(words) > 1:
                    if is_extended:
                        neg_map = _find_negation_scope(words)
                        for pos_idx, cue in neg_map.items():
                            target_w = words[pos_idx]
                            for neg_key in (f"neg:{target_w}", f"neg:{cue}_{target_w}",):
                                n_idx = self._hash_token(neg_key)
                                nb = 0.15 * table[n_idx] + 0.85 * c_vec
                                nb_norm = float(np.linalg.norm(nb))
                                table[n_idx] = nb / (nb_norm if nb_norm > 1e-12 else 1.0)
                        for pos_idx, w in enumerate(words):
                            if len(w) >= 3 and w not in _ANCHOR_STOPWORDS and pos_idx not in neg_map:
                                w_idx = self._hash_token(w)
                                if w_idx not in protected_word_indices:
                                    w_blended = 0.40 * table[w_idx] + 0.60 * c_vec
                                    wn = float(np.linalg.norm(w_blended))
                                    table[w_idx] = w_blended / (wn if wn > 1e-12 else 1.0)
                    else:
                        for w in words:
                            if len(w) >= 3:
                                w_idx = self._hash_token(w)
                                w_blended = 0.40 * table[w_idx] + 0.60 * c_vec
                                wn = float(np.linalg.norm(w_blended))
                                table[w_idx] = w_blended / (wn if wn > 1e-12 else 1.0)
                                protected_word_indices.add(w_idx)

        # Explicitly anchor canonical negated tokens (neg:<word>) to target centroids
        for target_cluster, pos_words in _NEGATED_CONCEPT_TARGETS.items():
            c_vec = anchor_centroids.get(target_cluster)
            if c_vec is None:
                continue
            for pw in pos_words:
                n_idx = self._hash_token(f"neg:{pw}")
                nb = 0.10 * table[n_idx] + 0.90 * c_vec
                nb_norm = float(np.linalg.norm(nb))
                table[n_idx] = nb / (nb_norm if nb_norm > 1e-12 else 1.0)
                for cue in ("not", "never", "no", "without", "cant", "cannot", "wont", "isnt", "wasnt", "didnt", "dont", "doesnt"):
                    pair_idx = self._hash_token(f"neg:{cue}_{pw}")
                    pb = 0.12 * table[pair_idx] + 0.88 * c_vec
                    pb_norm = float(np.linalg.norm(pb))
                    table[pair_idx] = pb / (pb_norm if pb_norm > 1e-12 else 1.0)

        return table

    def _init_embedding_table(self) -> np.ndarray:
        """Constructs the dense subword embedding table with calibrated semantic anchors."""
        return self._build_default_table()

    def _extract_bucket_weights(
        self,
        text: str,
        *,
        _pretokenized: Optional[Tuple[str, List[str], Dict[int, str]]] = None,
    ) -> Tuple[List[int], List[float]]:
        """Extracts weighted subword/phrase hash bucket indices for a single text."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        if not text or not text.strip():
            return [], []

        if _pretokenized is not None:
            clean_text, words, negated_positions = _pretokenized
        else:
            clean_text = _normalize_text(text)
            words, negated_positions = _tokenize_with_negation_scope(clean_text)

        indices: List[int] = []
        weights: List[float] = []
        hash_token = self._hash_token

        if " " in clean_text:
            seen_k3: set[str] = set()
            pos_sp = clean_text.find(" ")
            while pos_sp != -1:
                if 0 < pos_sp < len(clean_text) - 1:
                    k3 = clean_text[pos_sp - 1 : pos_sp + 2]
                    if k3 not in seen_k3:
                        seen_k3.add(k3)
                        cands = self._multiword_by_space3.get(k3)
                        if cands is not None:
                            for term, first_is_neg, term_hash in cands:
                                if term in clean_text:
                                    if not first_is_neg:
                                        pos_match = clean_text.find(term)
                                        clause_prefix = _CLAUSE_SPLIT_RE.split(clean_text[:pos_match])[-1]
                                        prefix_words = [
                                            w
                                            for w in re.findall(r"\b\w+\b", clause_prefix)
                                            if w not in _NEGATION_SKIP_MODIFIERS
                                        ]
                                        if prefix_words and prefix_words[-1] in _NEGATION_TOKENS:
                                            indices.append(term_hash)
                                            weights.append(-2.5)
                                            continue
                                    indices.append(term_hash)
                                    weights.append(3.5)
                pos_sp = clean_text.find(" ", pos_sp + 1)

        for pos, word in enumerate(words):
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            if pos in negated_positions:
                cue = negated_positions[pos]
                indices.append(hash_token(f"neg:{word}"))
                weights.append(word_weight * 2.8)
                indices.append(hash_token(f"neg:{cue}_{word}"))
                weights.append(word_weight * 2.0)
                indices.append(hash_token(word))
                weights.append(-word_weight * 0.9)
                continue

            if word in _NEGATION_TOKENS:
                indices.append(hash_token(word))
                weights.append(word_weight * 0.6)
                continue

            indices.append(hash_token(word))
            weights.append(word_weight * 1.5)

            wlen = len(word)
            if wlen >= 3:
                for n in (3, 4):
                    for j in range(wlen - n + 1):
                        sub = word[j : j + n]
                        indices.append(hash_token(f"sub:{sub}"))
                        weights.append(0.35)

        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            indices.append(hash_token(f"bi:{bigram}"))
            weights.append(1.2)

        return indices, weights

    def encode(
        self,
        text: str,
        *,
        _pretokenized: Optional[Tuple[str, List[str], Dict[int, str]]] = None,
    ) -> np.ndarray:
        """Encodes text into a normalized dense subword vector with negation-scope binding."""
        indices, weights = self._extract_bucket_weights(text, _pretokenized=_pretokenized)
        if not indices:
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        idx_arr = np.array(indices, dtype=np.int32)
        w_arr = np.array(weights, dtype=np.float32)

        accum = w_arr @ self._table[idx_arr]

        norm = float(np.linalg.norm(accum))
        if norm < 1e-12 or not np.isfinite(norm):
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v
        return (accum / norm).astype(np.float32)

    project = encode

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        _pretokenized_list: Optional[Sequence[Tuple[str, List[str], Dict[int, str]]]] = None,
    ) -> np.ndarray:
        """Batched encoding of multiple text inputs using chunked single-SGEMM unique-bucket lookup."""
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        n_docs = len(texts)
        if n_docs == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        if _pretokenized_list is not None and len(_pretokenized_list) != n_docs:
            raise ValueError("_pretokenized_list length must match texts length")

        out = np.empty((n_docs, self.dimension), dtype=np.float32)
        chunk_size = 256
        for start in range(0, n_docs, chunk_size):
            end = min(start + chunk_size, n_docs)
            c_len = end - start
            unique_buckets: Dict[int, int] = {}
            doc_maps: List[Dict[int, float]] = []
            empty_rows: List[int] = []

            for local_i, global_i in enumerate(range(start, end)):
                pretok = _pretokenized_list[global_i] if _pretokenized_list is not None else None
                idxs, wts = self._extract_bucket_weights(texts[global_i], _pretokenized=pretok)
                if not idxs:
                    doc_maps.append({})
                    empty_rows.append(local_i)
                    continue
                bmap: Dict[int, float] = {}
                for b_id, w in zip(idxs, wts):
                    if b_id not in unique_buckets:
                        unique_buckets[b_id] = len(unique_buckets)
                    bmap[b_id] = bmap.get(b_id, 0.0) + w
                doc_maps.append(bmap)

            if not unique_buckets:
                chunk = np.zeros((c_len, self.dimension), dtype=np.float32)
                chunk[:, 0] = 1.0
                out[start:end] = chunk
                continue

            bucket_ids = np.array(list(unique_buckets.keys()), dtype=np.int32)
            weight_mat = np.zeros((c_len, len(bucket_ids)), dtype=np.float32)
            for local_i, bmap in enumerate(doc_maps):
                if bmap:
                    cols = [unique_buckets[b_id] for b_id in bmap]
                    vals = list(bmap.values())
                    weight_mat[local_i, cols] = np.asarray(vals, dtype=np.float32)

            accums = weight_mat @ self._table[bucket_ids]
            norms = np.linalg.norm(accums, axis=1, keepdims=True)
            valid = (norms[:, 0] >= 1e-12) & np.isfinite(norms[:, 0])
            if empty_rows:
                valid[empty_rows] = False

            chunk = np.zeros_like(accums, dtype=np.float32)
            if np.any(valid):
                chunk[valid] = (accums[valid] / norms[valid]).astype(np.float32)
            if np.any(~valid):
                chunk[~valid, 0] = 1.0
            out[start:end] = chunk

        return out

    project_batch = encode_batch

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two text strings."""
        v_a = self.encode(text_a)
        v_b = self.encode(text_b)
        return float(np.dot(v_a, v_b))


StaticEmbeddingTable = SubwordSemanticEmbeddings


def _build_default_table(
    dimension: int = 256,
    vocab_size: int = 25000,
    seed: int = 42,
) -> np.ndarray:
    """Module-level helper to construct the default static concept-anchor subword table."""
    return SubwordSemanticEmbeddings(dimension=dimension, vocab_size=vocab_size, seed=seed)._table


def _extract_corpus_terms(text: str) -> List[str]:
    """Extract normalized word, stem, bigram, and negation-bound features for spectral fitting."""
    clean = _normalize_text(text)
    words, negated = _tokenize_with_negation_scope(clean, pattern=r"\b\w\w+\b", max_len=64)
    terms: List[str] = []
    for pos, w in enumerate(words):
        if pos in negated:
            cue = negated[pos]
            terms.append(f"neg:{w}")
            terms.append(f"{cue}_{w}")
        elif w not in _ANCHOR_STOPWORDS:
            terms.append(w)
            st = _stem_word(w)
            if st != w and len(st) >= 3:
                terms.append(f"stem:{st}")
    for i in range(len(words) - 1):
        if words[i] not in _ANCHOR_STOPWORDS or words[i + 1] not in _ANCHOR_STOPWORDS:
            terms.append(f"{words[i]}_{words[i + 1]}")
    return terms


def compute_strict_loo_weights(
    X_train: np.ndarray,
    Y_onehot: np.ndarray,
    *,
    hard_drop_thresh: float = 0.25,
    reg_loo: float = 0.75,
    min_samples: int = 40,
    min_per_class: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pure-NumPy closed-form Leave-One-Out ridge margin weights and clean sample mask."""
    n_samples, n_dim = X_train.shape
    n_classes = Y_onehot.shape[1]
    if n_samples < min_samples or n_classes < 2 or (n_samples / n_classes) < min_per_class:
        return np.ones(n_samples, dtype=np.float32), np.ones(n_samples, dtype=bool)

    x64 = np.asarray(X_train, dtype=np.float64)
    y64 = np.asarray(Y_onehot, dtype=np.float64)
    mu_y = np.mean(y64, axis=0)
    yc = y64 - mu_y
    eff_reg = max(1e-6, float(reg_loo))
    mu_x = np.mean(x64, axis=0)
    xc = x64 - mu_x

    if n_samples <= n_dim:
        gram = xc @ xc.T
        gram.flat[:: n_samples + 1] += eff_reg
        chol = np.linalg.cholesky(gram)
        chol_inv = np.linalg.solve(chol, np.eye(n_samples, dtype=np.float64))
        diag_kinv = np.sum(chol_inv * chol_inv, axis=0)
        kinv_yc = np.linalg.solve(chol.T, chol_inv @ yc)
        denom = np.maximum(1e-4, eff_reg * diag_kinv - (1.0 / n_samples))[:, None]
        y_loo = y64 - (eff_reg * kinv_yc) / denom
    else:
        u_mat, s_vals, _ = np.linalg.svd(xc, full_matrices=False)
        s2 = np.maximum(0.0, s_vals * s_vals)
        filt = s2 / (s2 + eff_reg)
        h_diag = (1.0 / n_samples) + (u_mat * u_mat) @ filt
        denom = np.maximum(1e-4, 1.0 - h_diag)[:, None]
        y_hat = mu_y + u_mat @ (filt[:, None] * (u_mat.T @ yc))
        y_loo = y64 + (y_hat - y64) / denom

    true_cls = np.argmax(y64, axis=1)
    s_true = y_loo[np.arange(n_samples), true_cls]
    rival_mask = (y64 <= 0.5) & (y_loo > mu_y[None, :])
    s_other = np.max(np.where(rival_mask, y_loo, -1e9), axis=1)
    deficit = s_other - s_true
    sw = np.where(
        deficit > 0.15,
        np.clip(1.0 - 2.6 * (deficit - 0.15), 0.05, 1.0),
        1.0,
    ).astype(np.float32)
    s_true_excess = s_true - mu_y[true_cls]
    sw = np.where(s_true_excess > 0.105, np.maximum(sw, 0.75), sw).astype(np.float32)
    for k in range(n_classes):
        idx = np.where(true_cls == k)[0]
        if len(idx) > 0:
            best = int(idx[int(np.argmin(deficit[idx]))])
            sw[best] = max(float(sw[best]), 0.85)
    clean_mask = sw >= float(hard_drop_thresh)
    return sw, clean_mask


def extract_schema_anchor_texts(schema: Any) -> List[str]:
    """Extracts human-readable semantic anchor phrases from a DecisionSchema-like object."""
    fields = getattr(schema, "fields", None)
    if not isinstance(fields, Mapping):
        return []
    anchors: List[str] = []
    for field_name, field in fields.items():
        options = getattr(field, "options", None)
        descriptions = getattr(field, "descriptions", None) or {}
        if options is not None:
            for opt in options:
                opt_str = str(opt)
                desc = str(descriptions.get(opt_str, "")).strip()
                phrase = f"{opt_str.replace('_', ' ')} {desc}".strip()
                if phrase:
                    anchors.append(phrase)
        else:
            desc = str(getattr(field, "description", "")).strip()
            phrase = f"{str(field_name).replace('_', ' ')} {desc}".strip()
            if phrase:
                anchors.append(phrase)
    return anchors


class _HybridFitDescriptor:
    """Allows HybridProjector.fit(...) to be called as either a classmethod or an instance method."""

    def __get__(self, instance: Optional[HybridProjector], owner: type[HybridProjector]) -> Any:
        if instance is None:
            return owner._fit_class
        return instance._fit_instance


class HybridProjector:
    """Concatenate weighted lexical TF-IDF and dense subword + PPMI/SVD spectral features.

    For unit components, square-root weights preserve unit norm; the
    implementation also normalizes its final vector. Unfitted instances default to
    1024 hashed lexical + 256 dense (alpha=0.5) for backward compatibility, while
    ``HybridProjector.fit(...)`` builds the unified v1.1 representation: exact
    ``TfidfProjector`` lexical features (default max 8000, char n-grams (3, 5)) +
    192-d dense subword embeddings blended with unsupervised PPMI+SVD spectral
    co-occurrence features (alpha=0.90).
    """

    CONFIG_KEYS = frozenset({
        "type",
        "sparse_dim",
        "dense_dim",
        "alpha",
        "seed",
        "fitted",
        "spectral_terms",
        "spectral_weights",
        "spectral_vectors",
        "subspace_components",
        "subspace_weight",
        "tfidf_terms",
        "tfidf_idf",
        "tfidf_char_ngrams",
        "svd_power",
        "spectral_blend",
        "use_subspace",
        "direct_spectral",
        "direct_spectral_dim",
        "context_smooth",
        "tfidf_use_stems",
        "tfidf_stem_weight",
        "tfidf_use_skip_bigrams",
        "tfidf_skip_weight",
        "tfidf_word_quota_frac",
        "tfidf_char_weight",
        "rbf_landmarks",
        "rbf_gamma",
        "rbf_weight",
    })

    extract_schema_anchor_texts = staticmethod(extract_schema_anchor_texts)

    def __init__(
        self,
        dimension: Optional[int] = None,
        *,
        sparse_dim: int = 1024,
        dense_dim: int = 256,
        alpha: Optional[float] = None,
        seed: int = 42,
        backend: str = "auto",
        svd_power: float = 0.5,
        spectral_blend: float = 0.35,
    ) -> None:
        self._explicit_alpha = alpha is not None
        eff_alpha = 0.5 if alpha is None else float(alpha)
        if not (0.0 < eff_alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {eff_alpha}")

        if dimension is not None:
            if dimension <= 0:
                raise ValueError(f"dimension must be positive, got {dimension}")
            # Divide dimension between sparse and dense
            sparse_dim = max(1, int(dimension * 0.75))
            dense_dim = max(1, dimension - sparse_dim)
            self.dimension = sparse_dim + dense_dim
        else:
            if sparse_dim <= 0:
                raise ValueError(f"sparse_dim must be positive, got {sparse_dim}")
            if dense_dim <= 0:
                raise ValueError(f"dense_dim must be positive, got {dense_dim}")
            self.dimension = sparse_dim + dense_dim

        self.sparse_dim = sparse_dim
        self.dense_dim = dense_dim
        self.alpha = eff_alpha
        self.seed = seed
        self.svd_power = float(svd_power)
        self.spectral_blend = float(spectral_blend)
        self.backend = self._resolve_backend(backend)

        # Dense subword semantic engine
        self._dense_engine = SubwordSemanticEmbeddings(
            dimension=self.dense_dim,
            vocab_size=25000,
            seed=self.seed,
        )

        # Optional corpus-fitted lexical TF-IDF + spectral + prototype projection state
        self._fitted: bool = False
        self._tfidf_projector: Optional[Any] = None
        self._spectral_terms: Tuple[str, ...] = ()
        self._spectral_vocab: Dict[str, int] = {}
        self._spectral_weights: Tuple[float, ...] = ()
        self._spectral_weights_arr: np.ndarray = np.empty(0, dtype=np.float32)
        self._spectral_vectors: Optional[np.ndarray] = None  # shape: (V, R_spec)
        self._spectral_basis: Optional[np.ndarray] = None    # shape: (R_spec, dense_dim)
        self._subspace_components: Optional[np.ndarray] = None  # shape: (R_sub, base_dim)
        self._subspace_weight: float = 0.0
        self._use_subspace: bool = False
        self._direct_spectral: bool = False
        self._direct_spectral_dim: int = 0
        self._context_smooth: float = 1.0
        self._rbf_landmarks: Optional[np.ndarray] = None
        self._rbf_gamma: float = 3.0
        self._rbf_weight: float = 0.0
        self._digest: Optional[str] = None

    @property
    def _base_dimension(self) -> int:
        return self.sparse_dim + self.dense_dim + (self._direct_spectral_dim if self._direct_spectral else 0)

    def _update_dimension(self) -> None:
        rbf_k = int(self._rbf_landmarks.shape[0]) if self._rbf_landmarks is not None else 0
        self.dimension = self._base_dimension + rbf_k

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def spectral_vocab(self) -> Dict[str, int]:
        return dict(self._spectral_vocab)

    @property
    def spectral_proj(self) -> Optional[np.ndarray]:
        if self._spectral_vectors is None:
            return None
        if self._spectral_basis is None:
            return self._spectral_vectors.astype(np.float32)
        return (self._spectral_vectors @ self._spectral_basis).astype(np.float32)

    def _resolve_backend(self, backend: str) -> str:
        if backend == "auto":
            return "mlx" if HAS_MLX else "numpy"
        if backend == "mlx":
            if not HAS_MLX:
                raise RuntimeError("MLX requested but mlx is not installed on this system")
            return "mlx"
        return "numpy"

    def _build_spectral_basis(self, r_spec: int) -> np.ndarray:
        """Deterministically generate an orthonormal basis (R_spec, dense_dim) from seed."""
        rng = np.random.RandomState(self.seed + 7919)
        mat = rng.normal(0.0, 1.0, size=(r_spec, self.dense_dim)).astype(np.float32)
        if r_spec <= self.dense_dim:
            q, _ = np.linalg.qr(mat.T)
            return q.T[:r_spec].astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return (mat / norms).astype(np.float32)

    def _fit_instance(
        self,
        texts: Iterable[str],
        labels: Optional[Sequence[Any]] = None,
        *,
        alpha: Optional[float] = None,
        max_vocab: int = 512,
        spectral_dim: int = 32,
        subspace_dim: int = 16,
        subspace_weight: float = 0.20,
        min_df: int = 1,
        ngram_range: Optional[Tuple[int, int]] = None,
        char_ngrams: Optional[Tuple[int, int]] = (3, 5),
        selection: str = "frequency",
        svd_power: Optional[float] = None,
        spectral_blend: Optional[float] = None,
        use_subspace: bool = False,
        direct_spectral: bool = False,
        context_smooth: float = 1.0,
        extra_texts: Sequence[str] = (),
        tfidf_use_stems: bool = False,
        tfidf_stem_weight: Optional[float] = None,
        tfidf_use_skip_bigrams: bool = False,
        tfidf_skip_weight: Optional[float] = None,
        tfidf_word_quota_frac: Optional[float] = None,
        tfidf_char_weight: Optional[float] = None,
        _auto_sparse_dim: bool = False,
        **kwargs: Any,
    ) -> HybridProjector:
        """Fit exact TF-IDF lexical channel, unsupervised PPMI/SVD spectral co-occurrence, and optional subspace."""
        _ = ngram_range
        if kwargs:
            unexpected = next(iter(kwargs))
            raise TypeError(f"HybridProjector.fit() got an unexpected keyword argument {unexpected!r}")
        if isinstance(texts, (str, bytes)) or texts is None:
            raise TypeError("texts must be a collection of strings")
        text_list = list(texts)
        for idx, t in enumerate(text_list):
            if not isinstance(t, str):
                raise TypeError(f"Expected string at index {idx}, got {type(t).__name__}")

        if alpha is not None:
            eff_alpha = float(alpha)
            if not (0.0 < eff_alpha < 1.0):
                raise ValueError(f"alpha must be in (0.0, 1.0), got {eff_alpha}")
            self.alpha = eff_alpha
            self._explicit_alpha = True
        elif not self._explicit_alpha:
            self.alpha = 0.90

        if svd_power is not None:
            self.svd_power = float(svd_power)
        if spectral_blend is not None:
            self.spectral_blend = float(spectral_blend)
        self._use_subspace = bool(use_subspace)
        self._direct_spectral = bool(direct_spectral)
        self._direct_spectral_dim = 0
        self._context_smooth = float(context_smooth)
        self._rbf_landmarks = None
        self._rbf_weight = 0.0

        label_list: Optional[List[str]] = None
        if labels is not None:
            if isinstance(labels, (str, bytes)):
                raise TypeError("labels must be a sequence matching texts")
            raw_labels = list(labels)
            if len(raw_labels) != len(text_list):
                raise ValueError("labels must have the same length as texts")
            label_list = [str(lbl) for lbl in raw_labels]

        self._digest = None
        self._fit_base_cache = {}
        self._tfidf_projector = None
        self._spectral_basis = None
        if not text_list and not extra_texts:
            self._fitted = True
            self._update_dimension()
            return self

        from system1.core.text import TfidfProjector

        extra_list = [str(t) for t in extra_texts if isinstance(t, str) and t.strip()]
        if extra_list and selection not in ("chi2", "mi"):
            tfidf_corpus = text_list + list(dict.fromkeys(extra_list))
            tfidf_labels = None
        else:
            tfidf_corpus = text_list
            tfidf_labels = label_list if selection in ("chi2", "mi") else None

        try:
            tfidf_kwargs: Dict[str, Any] = {
                "max_features": self.sparse_dim,
                "labels": tfidf_labels,
                "selection": selection,
                "char_ngrams": char_ngrams,
            }
            if tfidf_use_stems:
                tfidf_kwargs["use_stems"] = True
            if tfidf_stem_weight is not None:
                tfidf_kwargs["stem_weight"] = float(tfidf_stem_weight)
            if tfidf_use_skip_bigrams:
                tfidf_kwargs["use_skip_bigrams"] = True
            if tfidf_skip_weight is not None:
                tfidf_kwargs["skip_weight"] = float(tfidf_skip_weight)
            if tfidf_word_quota_frac is not None:
                tfidf_kwargs["word_quota_frac"] = float(tfidf_word_quota_frac)
            if tfidf_char_weight is not None:
                tfidf_kwargs["char_weight"] = float(tfidf_char_weight)
            tfidf = TfidfProjector.fit(tfidf_corpus, **tfidf_kwargs)
        except ValueError:
            tfidf = None

        self._tfidf_projector = tfidf
        if tfidf is not None and _auto_sparse_dim and tfidf.dimension < self.sparse_dim:
            self.sparse_dim = max(1, tfidf.dimension)
        self._update_dimension()

        spectral_corpus = text_list + extra_list
        n_docs = len(spectral_corpus)
        doc_terms: List[List[str]] = []
        df_counter: Counter[str] = Counter()
        tf_counter: Counter[str] = Counter()

        for t in spectral_corpus:
            terms = _extract_corpus_terms(t)
            doc_terms.append(terms)
            tf_counter.update(terms)
            df_counter.update(set(terms))

        if not tf_counter:
            self._fitted = True
            self._update_dimension()
            return self

        eff_min_df = max(1, int(min_df))
        candidate_terms = [tm for tm, df_int in df_counter.items() if df_int >= eff_min_df]
        if not candidate_terms:
            candidate_terms = list(df_counter.keys())

        term_scores: Dict[str, float] = {}
        term_weights_map: Dict[str, float] = {}
        for term in candidate_terms:
            df_int = df_counter[term]
            idf = math.log((1.0 + n_docs) / (1.0 + df_int)) + 1.0
            df_penalty = 0.25 if df_int == 1 and n_docs >= 40 else 1.0
            term_weights_map[term] = float(idf)
            term_scores[term] = float(math.log1p(tf_counter[term]) * idf * df_penalty)

        vocab_cap = 2048 if self._direct_spectral else 1024
        v_limit = max(1, min(int(max_vocab), vocab_cap))
        selected_vocab = sorted(
            sorted(term_scores.keys(), key=lambda tm: (-round(term_scores[tm], 9), -tf_counter[tm], tm))[:v_limit]
        )
        v_size = len(selected_vocab)
        vocab_idx = {tm: i for i, tm in enumerate(selected_vocab)}

        cooc = np.zeros((v_size, v_size), dtype=np.float64)
        chunk_docs = 1024
        for start in range(0, n_docs, chunk_docs):
            end = min(start + chunk_docs, n_docs)
            indicator = np.zeros((end - start, v_size), dtype=np.float64)
            for row_i, terms in enumerate(doc_terms[start:end]):
                present = {vocab_idx[tm] for tm in terms if tm in vocab_idx}
                if present:
                    indicator[row_i, list(present)] = 1.0
            cooc += indicator.T @ indicator

        total_cooc = float(np.sum(cooc))
        if total_cooc > 0.0:
            row_sums = np.sum(cooc, axis=1, keepdims=True)
            col_sums = np.sum(cooc, axis=0, keepdims=True)
            if self._context_smooth != 1.0:
                col_s_sm = np.power(np.maximum(col_sums, 1e-12), self._context_smooth)
                sm_sum = float(np.sum(col_s_sm))
                if sm_sum > 0.0:
                    col_sums = col_s_sm * (float(np.sum(col_sums)) / sm_sum)
            denom = np.maximum(row_sums @ col_sums, 1e-12)
            ratio = (cooc * total_cooc) / denom
            ppmi = np.maximum(0.0, np.log(np.maximum(ratio, 1e-12)))
        else:
            ppmi = np.eye(v_size, dtype=np.float64)

        if self._direct_spectral:
            r_spec = max(1, min(int(spectral_dim), v_size))
        else:
            r_spec = max(1, min(int(spectral_dim), v_size, self.dense_dim))
        u_mat, s_vals, _ = np.linalg.svd(ppmi, full_matrices=False)
        spec_vecs = (u_mat[:, :r_spec] * np.power(np.maximum(s_vals[:r_spec], 1e-12), self.svd_power)).astype(np.float32)
        spec_norms = np.linalg.norm(spec_vecs, axis=1, keepdims=True)
        spec_norms = np.where(spec_norms < 1e-12, 1.0, spec_norms)
        spec_vecs = (spec_vecs / spec_norms).astype(np.float32)
        spec_f16 = spec_vecs.astype(np.float16)
        if np.all(np.isfinite(spec_f16)):
            spec_vecs = spec_f16.astype(np.float32)

        self._spectral_terms = tuple(selected_vocab)
        self._spectral_vocab = vocab_idx
        self._spectral_weights = tuple(term_weights_map[tm] for tm in selected_vocab)
        self._spectral_weights_arr = np.asarray(self._spectral_weights, dtype=np.float32)
        self._spectral_vectors = spec_vecs

        if self._direct_spectral:
            self._direct_spectral_dim = r_spec
            self._spectral_basis = None
        else:
            vocab_dense = self._dense_engine.encode_batch(selected_vocab).astype(np.float32)
            aligned_basis = (spec_vecs.T @ vocab_dense).astype(np.float32)
            b_norms = np.linalg.norm(aligned_basis, axis=1, keepdims=True)
            if np.all(b_norms.flatten() > 1e-6):
                self._spectral_basis = (aligned_basis / b_norms).astype(np.float32)
            else:
                self._spectral_basis = self._build_spectral_basis(r_spec)

        self._update_dimension()
        self._subspace_components = None
        self._subspace_weight = float(subspace_weight) if self._use_subspace else 0.0
        self._fitted = True

        if self._use_subspace and len(text_list) >= 2:
            base_embs = self._project_base_batch(text_list).astype(np.float64)
            if len(text_list) <= 4096:
                self._fit_base_cache = {
                    t: base_embs[i].astype(np.float32) for i, t in enumerate(text_list)
                }
            classes = sorted(set(label_list)) if label_list is not None else []
            if label_list is not None and len(classes) >= 2:
                centroids: List[np.ndarray] = []
                for c in classes:
                    c_mask = [i for i, lbl in enumerate(label_list) if lbl == c]
                    if c_mask:
                        c_mean = np.mean(base_embs[c_mask], axis=0)
                        c_norm = float(np.linalg.norm(c_mean))
                        if c_norm > 1e-12:
                            centroids.append(c_mean / c_norm)
                if len(centroids) >= 2:
                    proto_mat = np.stack(centroids, axis=0)
                    centered_protos = proto_mat - np.mean(proto_mat, axis=0, keepdims=True)
                    _, s_p, vt_p = np.linalg.svd(centered_protos, full_matrices=False)
                    r_sub = max(1, min(int(subspace_dim), vt_p.shape[0], self._base_dimension))
                    valid = s_p[:r_sub] > 1e-6
                    if np.any(valid):
                        sub_arr = vt_p[:r_sub][valid].astype(np.float32)
                        sub_f16 = sub_arr.astype(np.float16)
                        self._subspace_components = sub_f16.astype(np.float32) if np.all(np.isfinite(sub_f16)) else sub_arr
            else:
                centered = base_embs - np.mean(base_embs, axis=0, keepdims=True)
                _, s_u, vt_u = np.linalg.svd(centered, full_matrices=False)
                r_sub = max(1, min(int(subspace_dim), vt_u.shape[0], self._base_dimension))
                valid = s_u[:r_sub] > 1e-6
                if np.any(valid):
                    sub_arr = vt_u[:r_sub][valid].astype(np.float32)
                    sub_f16 = sub_arr.astype(np.float16)
                    self._subspace_components = sub_f16.astype(np.float32) if np.all(np.isfinite(sub_f16)) else sub_arr

        self._digest = None
        return self

    @classmethod
    def _fit_class(
        cls,
        texts: Iterable[str],
        labels: Optional[Sequence[Any]] = None,
        *,
        dimension: Optional[int] = None,
        sparse_dim: Optional[int] = None,
        max_features: Optional[int] = None,
        dense_dim: int = 192,
        alpha: Optional[float] = None,
        seed: int = 42,
        backend: str = "auto",
        max_vocab: int = 512,
        spectral_dim: int = 32,
        subspace_dim: int = 16,
        subspace_weight: float = 0.20,
        min_df: int = 1,
        ngram_range: Optional[Tuple[int, int]] = None,
        char_ngrams: Optional[Tuple[int, int]] = (3, 5),
        selection: str = "frequency",
        svd_power: float = 0.5,
        spectral_blend: float = 0.35,
        use_subspace: bool = False,
        direct_spectral: bool = False,
        context_smooth: float = 1.0,
        extra_texts: Sequence[str] = (),
        tfidf_use_stems: bool = False,
        tfidf_stem_weight: Optional[float] = None,
        tfidf_use_skip_bigrams: bool = False,
        tfidf_skip_weight: Optional[float] = None,
        tfidf_word_quota_frac: Optional[float] = None,
        tfidf_char_weight: Optional[float] = None,
        **kwargs: Any,
    ) -> HybridProjector:
        auto_sparse_dim = dimension is None and sparse_dim is None and max_features is None
        eff_sparse_dim = (
            int(max_features)
            if max_features is not None
            else (int(sparse_dim) if sparse_dim is not None else 8000)
        )
        eff_alpha = 0.90 if alpha is None else float(alpha)
        projector = cls(
            dimension=dimension,
            sparse_dim=eff_sparse_dim,
            dense_dim=dense_dim,
            alpha=eff_alpha,
            seed=seed,
            backend=backend,
            svd_power=svd_power,
            spectral_blend=spectral_blend,
        )
        return projector._fit_instance(
            texts,
            labels=labels,
            alpha=eff_alpha,
            max_vocab=max_vocab,
            spectral_dim=spectral_dim,
            subspace_dim=subspace_dim,
            subspace_weight=subspace_weight,
            min_df=min_df,
            ngram_range=ngram_range,
            char_ngrams=char_ngrams,
            selection=selection,
            svd_power=svd_power,
            spectral_blend=spectral_blend,
            use_subspace=use_subspace,
            direct_spectral=direct_spectral,
            context_smooth=context_smooth,
            extra_texts=extra_texts,
            tfidf_use_stems=tfidf_use_stems,
            tfidf_stem_weight=tfidf_stem_weight,
            tfidf_use_skip_bigrams=tfidf_use_skip_bigrams,
            tfidf_skip_weight=tfidf_skip_weight,
            tfidf_word_quota_frac=tfidf_word_quota_frac,
            tfidf_char_weight=tfidf_char_weight,
            _auto_sparse_dim=auto_sparse_dim,
            **kwargs,
        )

    @classmethod
    def fit_v1_2(
        cls,
        texts: Iterable[str],
        labels: Optional[Sequence[Any]] = None,
        *,
        schema: Optional[Any] = None,
        schema_repeat: int = 4,
        dimension: Optional[int] = None,
        sparse_dim: Optional[int] = None,
        max_features: Optional[int] = None,
        dense_dim: int = 192,
        alpha: float = 0.88,
        seed: int = 42,
        backend: str = "auto",
        max_vocab: int = 2048,
        spectral_dim: int = 128,
        spectral_blend: float = 0.55,
        svd_power: float = 0.5,
        context_smooth: float = 0.75,
        char_ngrams: Optional[Tuple[int, int]] = (2, 5),
        tfidf_use_stems: bool = True,
        tfidf_stem_weight: float = 0.90,
        tfidf_use_skip_bigrams: bool = True,
        tfidf_skip_weight: float = 0.75,
        tfidf_word_quota_frac: float = 0.58,
        tfidf_char_weight: float = 0.42,
        direct_spectral: bool = True,
        rbf_gamma: float = 3.0,
        rbf_weight: float = 0.14,
        attach_rbf: bool = False,
        **kwargs: Any,
    ) -> HybridProjector:
        """Fit the v1.2 unified hybrid projector (rich lexical + schema-distributional spectral + optional RBF landmarks)."""
        text_list = list(texts)
        extra_texts: List[str] = []
        if schema is not None:
            anchors = extract_schema_anchor_texts(schema)
            if anchors and schema_repeat > 0:
                extra_texts = anchors * int(schema_repeat)
        proj = cls._fit_class(
            text_list,
            labels=labels,
            dimension=dimension,
            sparse_dim=sparse_dim,
            max_features=max_features,
            dense_dim=dense_dim,
            alpha=alpha,
            seed=seed,
            backend=backend,
            max_vocab=max_vocab,
            spectral_dim=spectral_dim,
            spectral_blend=spectral_blend,
            svd_power=svd_power,
            char_ngrams=char_ngrams,
            direct_spectral=direct_spectral,
            context_smooth=context_smooth,
            extra_texts=extra_texts,
            tfidf_use_stems=tfidf_use_stems,
            tfidf_stem_weight=tfidf_stem_weight,
            tfidf_use_skip_bigrams=tfidf_use_skip_bigrams,
            tfidf_skip_weight=tfidf_skip_weight,
            tfidf_word_quota_frac=tfidf_word_quota_frac,
            tfidf_char_weight=tfidf_char_weight,
            **kwargs,
        )
        if attach_rbf and labels is not None and len(text_list) >= 2:
            proj.attach_rbf_landmarks(
                text_list,
                labels,
                rbf_gamma=rbf_gamma,
                rbf_weight=rbf_weight,
            )
        return proj

    def attach_rbf_landmarks(
        self,
        texts: Sequence[str],
        labels: Sequence[Any],
        *,
        rbf_gamma: float = 3.0,
        rbf_weight: float = 0.14,
        hard_drop_thresh: float = 0.35,
        reg_loo: float = 0.75,
    ) -> HybridProjector:
        """Attach LOO-filtered class-centroid RBF-Cosine similarity landmarks on base features."""
        text_list = list(texts)
        label_list = [str(lbl) for lbl in labels]
        if len(text_list) != len(label_list):
            raise ValueError("labels must match texts length")
        if not text_list:
            raise ValueError("texts must be non-empty to attach RBF landmarks")

        eff_gamma = float(rbf_gamma)
        eff_weight = float(rbf_weight)
        if eff_gamma <= 0.0 or not math.isfinite(eff_gamma):
            raise ValueError(f"rbf_gamma must be positive and finite, got {rbf_gamma}")
        if not (0.0 < eff_weight < 1.0) or not math.isfinite(eff_weight):
            raise ValueError(f"rbf_weight must be in (0, 1), got {rbf_weight}")

        self._rbf_landmarks = None
        self._rbf_weight = 0.0
        self._update_dimension()

        base_x = self.project_batch(text_list).astype(np.float64)
        classes = sorted(set(label_list))
        if len(classes) < 2:
            return self
        c_map = {c: i for i, c in enumerate(classes)}
        n_samples = len(text_list)
        y_mat = np.zeros((n_samples, len(classes)), dtype=np.float64)
        for i, lbl in enumerate(label_list):
            y_mat[i, c_map[lbl]] = 1.0

        sw, clean_mask = compute_strict_loo_weights(
            base_x,
            y_mat,
            hard_drop_thresh=hard_drop_thresh,
            reg_loo=reg_loo,
        )
        if int(np.sum(clean_mask)) >= len(classes):
            xc = base_x[clean_mask]
            yc = y_mat[clean_mask]
            swc = sw[clean_mask]
        else:
            xc, yc, swc = base_x, y_mat, sw

        landmarks: List[np.ndarray] = []
        for k in range(len(classes)):
            mask = yc[:, k] > 0.5
            if np.any(mask):
                w_k = swc[mask][:, None]
                m_k = np.sum(xc[mask] * w_k, axis=0) / max(1e-8, float(np.sum(w_k)))
                mn = float(np.linalg.norm(m_k))
                landmarks.append((m_k / (mn if mn > 1e-12 else 1.0)).astype(np.float32))
        if landmarks:
            lm_arr = np.stack(landmarks, axis=0).astype(np.float32)
            lm_f16 = lm_arr.astype(np.float16)
            if np.all(np.isfinite(lm_f16)):
                lm_arr = lm_f16.astype(np.float32)
            self._rbf_landmarks = lm_arr
            self._rbf_gamma = eff_gamma
            self._rbf_weight = eff_weight
            self._update_dimension()
            self._digest = None
        return self

    fit = _HybridFitDescriptor()

    @staticmethod
    def _sparse_word_hashes(token: str, sparse_dim: int, count: int = 4) -> Tuple[Tuple[int, float], ...]:
        return _sparse_word_hashes_cached(token, sparse_dim, count)

    @staticmethod
    def _sparse_single_hash(prefix_token: str, sparse_dim: int) -> Tuple[int, float]:
        return _sparse_single_hash_cached(prefix_token, sparse_dim)

    def _hash_to_sparse_vec(
        self,
        text: str,
        *,
        _pretokenized: Optional[Tuple[str, List[str], Dict[int, str]]] = None,
    ) -> np.ndarray:
        """Projects text into normalized sparse lexical n-gram vector with negation-scope binding."""
        vec = np.zeros(self.sparse_dim, dtype=np.float32)
        if not text or not text.strip():
            vec[0] = 1.0
            return vec

        if _pretokenized is not None:
            clean_text, words, negated_positions = _pretokenized
        else:
            clean_text = _normalize_text(text)
            words, negated_positions = _tokenize_with_negation_scope(clean_text)

        for pos, word in enumerate(words):
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            if pos in negated_positions:
                cue = negated_positions[pos]
                for neg_token, mult in ((f"neg:{word}", 2.6), (f"neg:{cue}_{word}", 1.8)):
                    boost = 1.0
                    if self._fitted and neg_token in self._spectral_vocab:
                        boost = 1.0 + min(1.8, self._spectral_weights[self._spectral_vocab[neg_token]] * 0.22)
                    scaled_neg = word_weight * mult * boost
                    for idx, sign in self._sparse_word_hashes(neg_token, self.sparse_dim, 4):
                        vec[idx] += sign * scaled_neg
                continue

            is_stop = word in _ANCHOR_STOPWORDS
            corpus_boost = 1.0
            if self._fitted:
                if word in self._spectral_vocab:
                    corpus_boost = 1.0 + min(2.2, self._spectral_weights[self._spectral_vocab[word]] * 0.28)
                elif is_stop:
                    corpus_boost = 0.25

            scaled_word = word_weight * 2.0 * corpus_boost
            for idx, sign in self._sparse_word_hashes(word, self.sparse_dim, 4):
                vec[idx] += sign * scaled_word

            if self._fitted and not is_stop and len(word) >= 3:
                stem = _stem_word(word)
                if stem != word and len(stem) >= 3:
                    stem_key = f"stem:{stem}"
                    stem_boost = 1.0
                    if stem_key in self._spectral_vocab:
                        stem_boost = 1.0 + min(1.8, self._spectral_weights[self._spectral_vocab[stem_key]] * 0.22)
                    elif stem in self._spectral_vocab:
                        stem_boost = 1.0 + min(1.8, self._spectral_weights[self._spectral_vocab[stem]] * 0.22)
                    scaled_stem = word_weight * 1.15 * stem_boost
                    for idx, sign in self._sparse_word_hashes(stem_key, self.sparse_dim, 2):
                        vec[idx] += sign * scaled_stem

            if len(word) >= 3 and not (self._fitted and is_stop and word not in self._spectral_vocab):
                ng_source = f"<{word}>" if self._fitted else word
                ng_weight = 0.36 if self._fitted else 0.5
                for n in (3, 4):
                    for j in range(len(ng_source) - n + 1):
                        ngram = ng_source[j : j + n]
                        idx_ng, sign_ng = self._sparse_single_hash(f"ng:{ngram}", self.sparse_dim)
                        vec[idx_ng] += sign_ng * ng_weight

        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            bi_weight = 1.5
            if self._fitted:
                if bigram in self._spectral_vocab:
                    bi_weight = 1.5 * (1.0 + min(1.8, self._spectral_weights[self._spectral_vocab[bigram]] * 0.25))
                elif words[i] in _ANCHOR_STOPWORDS and words[i + 1] in _ANCHOR_STOPWORDS:
                    bi_weight = 0.30
            idx_bi, sign_bi = self._sparse_single_hash(f"bi:{bigram}", self.sparse_dim)
            vec[idx_bi] += sign_bi * bi_weight

        norm = float(np.linalg.norm(vec))
        if norm < 1e-12 or not np.isfinite(norm):
            v = np.zeros(self.sparse_dim, dtype=np.float32)
            v[0] = 1.0
            return v
        return (vec / norm).astype(np.float32)

    def _extract_spectral_unit(self, text: str) -> Tuple[Optional[np.ndarray], bool]:
        """Compute unit-normalized spectral vector for a single document, or (None, False)."""
        if (
            not self._fitted
            or self._spectral_vectors is None
            or not self._spectral_vocab
        ):
            return None, False
        terms = _extract_corpus_terms(text)
        if not terms:
            return None, False
        counts = Counter(t for t in terms if t in self._spectral_vocab)
        if not counts:
            return None, False
        r_spec = self._spectral_vectors.shape[1]
        spec_accum = np.zeros(r_spec, dtype=np.float32)
        for tm, cnt in counts.items():
            idx = self._spectral_vocab[tm]
            w = (1.0 + math.log(cnt)) * self._spectral_weights[idx]
            spec_accum += w * self._spectral_vectors[idx]
        s_norm = float(np.linalg.norm(spec_accum))
        if s_norm < 1e-12:
            return None, False
        return (spec_accum / s_norm).astype(np.float32), True

    def _extract_spectral_unit_batch(self, texts: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Batched unit-normalized raw spectral vectors and validity mask via chunked SGEMM."""
        n_docs = len(texts)
        if (
            not self._fitted
            or self._spectral_vectors is None
            or not self._spectral_vocab
        ):
            r_spec = self._spectral_vectors.shape[1] if self._spectral_vectors is not None else max(0, self._direct_spectral_dim)
            return np.zeros((n_docs, r_spec), dtype=np.float32), np.zeros(n_docs, dtype=bool)

        v_size, r_spec = self._spectral_vectors.shape
        sw_arr = self._spectral_weights_arr
        if len(sw_arr) != v_size:
            sw_arr = np.asarray(self._spectral_weights, dtype=np.float32)
            self._spectral_weights_arr = sw_arr
        sv = self._spectral_vocab

        out = np.zeros((n_docs, r_spec), dtype=np.float32)
        has_spec = np.zeros(n_docs, dtype=bool)
        chunk_size = 256
        for start in range(0, n_docs, chunk_size):
            end = min(start + chunk_size, n_docs)
            c_len = end - start
            counts_mat = np.zeros((c_len, v_size), dtype=np.float32)
            non_empty_rows: List[int] = []
            for local_i, t in enumerate(texts[start:end]):
                terms = _extract_corpus_terms(t)
                if not terms:
                    continue
                cnts: Dict[int, int] = {}
                for tm in terms:
                    idx = sv.get(tm)
                    if idx is not None:
                        cnts[idx] = cnts.get(idx, 0) + 1
                if cnts:
                    cols = list(cnts.keys())
                    vals = [
                        float(sw_arr[idx]) if c == 1 else (1.0 + math.log(c)) * float(sw_arr[idx])
                        for idx, c in cnts.items()
                    ]
                    counts_mat[local_i, cols] = np.asarray(vals, dtype=np.float32)
                    non_empty_rows.append(local_i)
            if not non_empty_rows:
                continue
            accums = counts_mat @ self._spectral_vectors
            norms = np.linalg.norm(accums, axis=1, keepdims=True)
            valid = (norms[:, 0] >= 1e-12) & np.isfinite(norms[:, 0])
            if np.any(valid):
                accums[valid] /= norms[valid]
                out[start:end][valid] = accums[valid].astype(np.float32)
                has_spec[start:end] = valid
        return out, has_spec

    def _encode_spectral_dense(self, text: str, base_dense: np.ndarray) -> np.ndarray:
        """Blends corpus-learned PPMI/SVD spectral co-occurrence vector into dense channel."""
        if self._spectral_basis is None or self.spectral_blend <= 0.0:
            return base_dense
        spec_unit, ok = self._extract_spectral_unit(text)
        if not ok or spec_unit is None:
            return base_dense
        spec_dense = spec_unit @ self._spectral_basis
        sd_norm = float(np.linalg.norm(spec_dense))
        if sd_norm > 1e-12:
            spec_dense = spec_dense / sd_norm
        blend = min(1.0, max(0.0, float(self.spectral_blend)))
        blended = (1.0 - blend) * base_dense + blend * spec_dense
        b_norm = float(np.linalg.norm(blended))
        return (blended / b_norm).astype(np.float32) if b_norm > 1e-12 else base_dense

    def _project_base(self, text: str) -> np.ndarray:
        fit_cache = getattr(self, "_fit_base_cache", None)
        if fit_cache and text in fit_cache:
            return fit_cache[text]
        if text and text.strip():
            clean_text = _normalize_text(text)
            pretok: Optional[Tuple[str, List[str], Dict[int, str]]] = (
                clean_text,
                *_tokenize_with_negation_scope(clean_text),
            )
        else:
            pretok = None
        if self._tfidf_projector is not None:
            tf_vec = self._tfidf_projector.project(text)
            if tf_vec.shape[0] == self.sparse_dim:
                v_sparse = tf_vec
            else:
                v_sparse = np.zeros(self.sparse_dim, dtype=np.float32)
                n_copy = min(self.sparse_dim, tf_vec.shape[0])
                v_sparse[:n_copy] = tf_vec[:n_copy]
        else:
            v_sparse = self._hash_to_sparse_vec(text, _pretokenized=pretok)
        v_dense = self._dense_engine.encode(text, _pretokenized=pretok)

        if self._direct_spectral:
            sd = self.sparse_dim
            dd = self.dense_dim
            spd = self._direct_spectral_dim
            base_dim = sd + dd + spd
            w_sparse = math.sqrt(self.alpha)
            if spd > 0:
                w_dense = math.sqrt((1.0 - self.alpha) * max(0.0, 1.0 - self.spectral_blend))
                w_spec = math.sqrt((1.0 - self.alpha) * max(0.0, self.spectral_blend))
            else:
                w_dense = math.sqrt(1.0 - self.alpha)
                w_spec = 0.0
            v_out = np.empty(base_dim, dtype=np.float32)
            np.multiply(v_sparse, w_sparse, out=v_out[:sd])
            np.multiply(v_dense, w_dense, out=v_out[sd : sd + dd])
            if spd > 0:
                spec_unit, ok = self._extract_spectral_unit(text)
                if ok and spec_unit is not None and spec_unit.shape[0] == spd:
                    np.multiply(spec_unit, w_spec, out=v_out[sd + dd :])
                else:
                    v_out[sd + dd :] = 0.0
            norm = float(np.linalg.norm(v_out))
            if norm > 1e-12 and np.isfinite(norm):
                return (v_out / norm).astype(np.float32)
            v_fallback = np.zeros(base_dim, dtype=np.float32)
            v_fallback[0] = 1.0
            return v_fallback

        if self._fitted:
            v_dense = self._encode_spectral_dense(text, v_dense)

        w_sparse = math.sqrt(self.alpha)
        w_dense = math.sqrt(1.0 - self.alpha)

        v_hybrid = np.concatenate([w_sparse * v_sparse, w_dense * v_dense]).astype(np.float32)
        norm = float(np.linalg.norm(v_hybrid))
        if norm > 1e-12:
            return (v_hybrid / norm).astype(np.float32)
        v_fallback = np.zeros(self._base_dimension, dtype=np.float32)
        v_fallback[0] = 1.0
        return v_fallback

    def _project_base_batch(
        self,
        texts: Sequence[str],
        *,
        return_norms: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Batched projection into base feature space prior to subspace/RBF stages."""
        n_docs = len(texts)
        base_dim = self._base_dimension
        if n_docs == 0:
            empty_x = np.empty((0, base_dim), dtype=np.float32)
            if return_norms:
                return empty_x, np.empty(0, dtype=np.float32)
            return empty_x

        if self._tfidf_projector is not None:
            pretok_list = None
            if return_norms:
                tf_vecs, raw_norms = self._tfidf_projector.project_batch(texts, return_norms=True)
            else:
                tf_vecs = self._tfidf_projector.project_batch(texts, return_norms=False)
                raw_norms = None
            if tf_vecs.shape[1] == self.sparse_dim:
                v_sparse_mat = tf_vecs
            else:
                v_sparse_mat = np.zeros((n_docs, self.sparse_dim), dtype=np.float32)
                n_copy = min(self.sparse_dim, tf_vecs.shape[1])
                v_sparse_mat[:, :n_copy] = tf_vecs[:, :n_copy]
        else:
            pretok_list = [
                (ct, *_tokenize_with_negation_scope(ct)) if (t and t.strip() and (ct := _normalize_text(t))) else ("", [], {})
                for t in texts
            ]
            v_sparse_mat = np.stack(
                [self._hash_to_sparse_vec(t, _pretokenized=pt if t and t.strip() else None) for t, pt in zip(texts, pretok_list)],
                axis=0,
            ).astype(np.float32)
            if return_norms:
                raw_norms = np.array([1.0 if t.strip() else 0.0 for t in texts], dtype=np.float32)
            else:
                raw_norms = None

        v_dense_mat = self._dense_engine.encode_batch(
            texts,
            _pretokenized_list=[
                pt if (t and t.strip()) else None for t, pt in zip(texts, pretok_list)
            ] if pretok_list is not None else None,
        )

        if self._direct_spectral:
            sd = self.sparse_dim
            dd = self.dense_dim
            spd = self._direct_spectral_dim
            w_sparse = math.sqrt(self.alpha)
            if spd > 0:
                w_dense = math.sqrt((1.0 - self.alpha) * max(0.0, 1.0 - self.spectral_blend))
                w_spec = math.sqrt((1.0 - self.alpha) * max(0.0, self.spectral_blend))
            else:
                w_dense = math.sqrt(1.0 - self.alpha)
                w_spec = 0.0

            base_mat = np.empty((n_docs, base_dim), dtype=np.float32)
            np.multiply(v_sparse_mat, w_sparse, out=base_mat[:, :sd])
            np.multiply(v_dense_mat, w_dense, out=base_mat[:, sd : sd + dd])
            if spd > 0:
                spec_mat, has_spec = self._extract_spectral_unit_batch(texts)
                if spec_mat.shape[1] == spd:
                    np.multiply(spec_mat, w_spec, out=base_mat[:, sd + dd :])
                    if np.any(~has_spec):
                        base_mat[~has_spec, sd + dd :] = 0.0
                else:
                    base_mat[:, sd + dd :] = 0.0
        else:
            if (
                self._fitted
                and self._spectral_vectors is not None
                and self._spectral_basis is not None
                and self._spectral_vocab
                and self.spectral_blend > 0.0
            ):
                spec_mat, has_spec = self._extract_spectral_unit_batch(texts)
                if np.any(has_spec):
                    spec_dense = spec_mat[has_spec] @ self._spectral_basis
                    sd_norms = np.linalg.norm(spec_dense, axis=1, keepdims=True)
                    good_sd = sd_norms[:, 0] > 1e-12
                    if np.any(good_sd):
                        spec_dense[good_sd] = spec_dense[good_sd] / sd_norms[good_sd]
                    blend = min(1.0, max(0.0, float(self.spectral_blend)))
                    blended = (1.0 - blend) * v_dense_mat[has_spec] + blend * spec_dense
                    b_norms = np.linalg.norm(blended, axis=1, keepdims=True)
                    good_b = b_norms[:, 0] > 1e-12
                    has_idx = np.flatnonzero(has_spec)
                    if np.any(good_b):
                        v_dense_mat[has_idx[good_b]] = (blended[good_b] / b_norms[good_b]).astype(np.float32)

            w_sparse = math.sqrt(self.alpha)
            w_dense = math.sqrt(1.0 - self.alpha)
            base_mat = np.empty((n_docs, base_dim), dtype=np.float32)
            np.multiply(v_sparse_mat, w_sparse, out=base_mat[:, : self.sparse_dim])
            np.multiply(v_dense_mat, w_dense, out=base_mat[:, self.sparse_dim :])

        norms = np.linalg.norm(base_mat, axis=1, keepdims=True)
        valid = (norms[:, 0] > 1e-12) & np.isfinite(norms[:, 0])
        if np.any(valid):
            base_mat[valid] = (base_mat[valid] / norms[valid]).astype(np.float32)
        if np.any(~valid):
            base_mat[~valid] = 0.0
            base_mat[~valid, 0] = 1.0

        if return_norms:
            assert raw_norms is not None
            return base_mat, raw_norms
        return base_mat

    def raw_norm(self, text: str) -> float:
        """Returns pre-normalization lexical magnitude (TF-IDF L2 norm when fitted with TF-IDF)."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        if self._tfidf_projector is not None:
            return float(self._tfidf_projector.raw_norm(text))
        return 1.0 if text.strip() else 0.0

    def project(self, text: str, recency_weighted: Optional[bool] = None) -> np.ndarray:
        """Projects input text into unit-normalized hybrid sparse-dense representation."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")

        v_base = self._project_base(text)
        if self._fitted and self._subspace_components is not None and self._subspace_weight > 0.0:
            coords = self._subspace_components @ v_base
            v_sub = self._subspace_components.T @ coords
            v_base = v_base + float(self._subspace_weight) * v_sub
            norm = float(np.linalg.norm(v_base))
            if norm > 1e-12 and np.isfinite(norm):
                v_base = (v_base / norm).astype(np.float32)
            else:
                v_base = np.zeros(self._base_dimension, dtype=np.float32)
                v_base[0] = 1.0

        if self._rbf_landmarks is not None and self._rbf_weight > 0.0:
            sims = self._rbf_landmarks @ v_base
            phi = np.exp(self._rbf_gamma * (sims - 1.0)).astype(np.float32)
            pnorm = float(np.linalg.norm(phi))
            wb = math.sqrt(1.0 - self._rbf_weight)
            wr = math.sqrt(self._rbf_weight)
            d_b = len(v_base)
            v_out = np.empty(self.dimension, dtype=np.float32)
            np.multiply(v_base, wb, out=v_out[:d_b])
            if pnorm > 1e-12:
                np.multiply(phi, wr / pnorm, out=v_out[d_b:])
                if not np.any(v_base):
                    on = float(np.linalg.norm(v_out))
                    if on > 1e-12:
                        v_out /= on
            else:
                v_out[d_b:] = 0.0
                on = float(np.linalg.norm(v_out))
                if on > 1e-12:
                    v_out /= on
            return v_out

        return v_base

    encode = project

    def __call__(self, text: str) -> np.ndarray:
        return self.project(text)

    def project_batch(
        self,
        texts: Sequence[str],
        recency_weighted: Optional[bool] = None,
        *,
        return_norms: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Batched projection of multiple text inputs using vectorized sparse, dense, and spectral stages."""
        _ = recency_weighted
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        text_list = list(texts) if not isinstance(texts, (list, tuple)) else texts
        if len(text_list) == 0:
            empty_x = np.empty((0, self.dimension), dtype=np.float32)
            if return_norms:
                return empty_x, np.empty(0, dtype=np.float32)
            return empty_x
        for idx, t in enumerate(text_list):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be strings, got {type(t).__name__} at index {idx}")

        if return_norms:
            v_base_mat, raw_norms = self._project_base_batch(text_list, return_norms=True)
        else:
            v_base_mat = self._project_base_batch(text_list, return_norms=False)
            raw_norms = None

        if self._fitted and self._subspace_components is not None and self._subspace_weight > 0.0:
            coords = v_base_mat @ self._subspace_components.T
            v_sub = coords @ self._subspace_components
            v_base_mat = (v_base_mat + float(self._subspace_weight) * v_sub).astype(np.float32)
            s_norms = np.linalg.norm(v_base_mat, axis=1, keepdims=True)
            valid = (s_norms[:, 0] > 1e-12) & np.isfinite(s_norms[:, 0])
            if np.any(valid):
                v_base_mat[valid] = (v_base_mat[valid] / s_norms[valid]).astype(np.float32)
            if np.any(~valid):
                v_base_mat[~valid] = 0.0
                v_base_mat[~valid, 0] = 1.0

        if self._rbf_landmarks is not None and self._rbf_weight > 0.0:
            sims = v_base_mat @ self._rbf_landmarks.T
            phi = np.exp(self._rbf_gamma * (sims - 1.0)).astype(np.float32)
            p_norms = np.linalg.norm(phi, axis=1, keepdims=True)
            phi_unit = np.where(p_norms > 1e-12, phi / np.maximum(p_norms, 1e-12), 0.0).astype(np.float32)
            wb = math.sqrt(1.0 - self._rbf_weight)
            wr = math.sqrt(self._rbf_weight)
            d_b = v_base_mat.shape[1]
            v_out = np.empty((len(text_list), self.dimension), dtype=np.float32)
            np.multiply(v_base_mat, wb, out=v_out[:, :d_b])
            np.multiply(phi_unit, wr, out=v_out[:, d_b:])
            row_norms = np.linalg.norm(v_out, axis=1, keepdims=True)
            valid_out = (row_norms[:, 0] > 1e-12) & np.isfinite(row_norms[:, 0])
            if np.any(valid_out):
                v_out[valid_out] = (v_out[valid_out] / row_norms[valid_out]).astype(np.float32)
            if np.any(~valid_out):
                v_out[~valid_out] = 0.0
                v_out[~valid_out, 0] = 1.0
            res = v_out
        else:
            res = v_base_mat

        if return_norms:
            assert raw_norms is not None
            return res, raw_norms
        return res

    encode_batch = project_batch

    def similarity(self, text_a: str, text_b: str) -> float:
        """Computes cosine similarity between two text inputs in hybrid space."""
        v_a = self.project(text_a)
        v_b = self.project(text_b)
        return float(np.dot(v_a, v_b))

    def to_config(self, *, include_arrays: bool = True) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "type": "hybrid",
            "sparse_dim": self.sparse_dim,
            "dense_dim": self.dense_dim,
            "alpha": self.alpha,
            "seed": self.seed,
        }
        if self._fitted:
            cfg["fitted"] = True
            cfg["svd_power"] = round(float(self.svd_power), 6)
            cfg["spectral_blend"] = round(float(self.spectral_blend), 6)
            cfg["spectral_terms"] = list(self._spectral_terms)
            cfg["spectral_weights"] = [round(float(w), 6) for w in self._spectral_weights]
            if self._direct_spectral:
                cfg["direct_spectral"] = True
                cfg["direct_spectral_dim"] = int(self._direct_spectral_dim)
            if abs(self._context_smooth - 1.0) > 1e-9:
                cfg["context_smooth"] = round(float(self._context_smooth), 6)
            if self._tfidf_projector is not None:
                tp_cfg = self._tfidf_projector.to_config(include_arrays=include_arrays)
                cfg["tfidf_terms"] = tp_cfg["terms"]
                if "char_ngrams" in tp_cfg and tp_cfg["char_ngrams"] is not None:
                    cfg["tfidf_char_ngrams"] = tp_cfg["char_ngrams"]
                for src_k, dst_k in (
                    ("use_stems", "tfidf_use_stems"),
                    ("stem_weight", "tfidf_stem_weight"),
                    ("use_skip_bigrams", "tfidf_use_skip_bigrams"),
                    ("skip_weight", "tfidf_skip_weight"),
                    ("word_quota_frac", "tfidf_word_quota_frac"),
                    ("char_weight", "tfidf_char_weight"),
                ):
                    if src_k in tp_cfg:
                        cfg[dst_k] = tp_cfg[src_k]
                if include_arrays and "idf" in tp_cfg:
                    cfg["tfidf_idf"] = tp_cfg["idf"]
            if self._use_subspace:
                cfg["use_subspace"] = True
            if self._rbf_landmarks is not None:
                cfg["rbf_gamma"] = round(float(self._rbf_gamma), 6)
                cfg["rbf_weight"] = round(float(self._rbf_weight), 6)
                if include_arrays:
                    cfg["rbf_landmarks"] = [
                        [round(float(x), 6) for x in row] for row in self._rbf_landmarks
                    ]
            if include_arrays:
                if self._spectral_vectors is not None:
                    cfg["spectral_vectors"] = [
                        [round(float(x), 6) for x in row] for row in self._spectral_vectors
                    ]
                if self._subspace_components is not None:
                    cfg["subspace_components"] = [
                        [round(float(x), 6) for x in row] for row in self._subspace_components
                    ]
            cfg["subspace_weight"] = round(float(self._subspace_weight), 6)
        return cfg

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        backend: str = "auto",
        spectral_vectors: Optional[np.ndarray] = None,
        subspace_components: Optional[np.ndarray] = None,
        tfidf_idf_array: Optional[np.ndarray] = None,
        rbf_landmarks: Optional[np.ndarray] = None,
    ) -> HybridProjector:
        if not isinstance(config, Mapping) or config.get("type") != "hybrid":
            raise ValueError("Unsupported hybrid projector settings")
        if set(config.keys()) - cls.CONFIG_KEYS:
            raise ValueError("Saved hybrid projector contains unsupported settings")
        sparse = config.get("sparse_dim")
        dense = config.get("dense_dim")
        if type(sparse) is not int or type(dense) is not int or sparse <= 0 or dense <= 0:
            raise ValueError("Saved hybrid projector dimensions do not match the skill")
        alpha = float(config.get("alpha", 0.5))
        seed = int(config.get("seed", 42))
        svd_power = float(config.get("svd_power", 0.5))
        spectral_blend = float(config.get("spectral_blend", 0.35))
        terms = tuple(config.get("spectral_terms", ()))
        if any(not isinstance(term, str) or not term.strip() for term in terms) or len(set(terms)) != len(terms):
            raise ValueError("Saved hybrid spectral terms must be unique nonempty strings")
        proj = cls(
            sparse_dim=sparse,
            dense_dim=dense,
            alpha=alpha,
            seed=seed,
            backend=backend,
            svd_power=svd_power,
            spectral_blend=spectral_blend,
        )
        proj._use_subspace = bool(config.get("use_subspace", False))
        proj._direct_spectral = bool(config.get("direct_spectral", False))
        proj._direct_spectral_dim = int(config.get("direct_spectral_dim", 0)) if proj._direct_spectral else 0
        proj._context_smooth = float(config.get("context_smooth", 1.0))

        _hybrid_tfidf_key_map = (
            ("tfidf_use_stems", "use_stems"),
            ("tfidf_stem_weight", "stem_weight"),
            ("tfidf_use_skip_bigrams", "use_skip_bigrams"),
            ("tfidf_skip_weight", "skip_weight"),
            ("tfidf_word_quota_frac", "word_quota_frac"),
            ("tfidf_char_weight", "char_weight"),
        )
        _hybrid_all_tfidf_keys = {
            "tfidf_terms",
            "tfidf_idf",
            "tfidf_char_ngrams",
            *(src_k for src_k, _ in _hybrid_tfidf_key_map),
        }
        has_any_tfidf_cfg = bool(set(config.keys()) & _hybrid_all_tfidf_keys)

        if config.get("fitted"):
            if "tfidf_terms" not in config:
                if has_any_tfidf_cfg or tfidf_idf_array is not None:
                    raise ValueError("Saved hybrid projector has incomplete TF-IDF configuration")
            else:
                from system1.core.text import DEFAULT_MAX_FEATURES, TfidfProjector

                tf_terms = config["tfidf_terms"]
                if not isinstance(tf_terms, (list, tuple)) or not 0 < len(tf_terms) <= sparse:
                    raise ValueError("Saved hybrid TF-IDF vocabulary exceeds sparse_dim")
                tf_cfg: Dict[str, Any] = {"type": "tfidf", "terms": list(tf_terms)}
                if "tfidf_char_ngrams" in config:
                    tf_cfg["char_ngrams"] = config["tfidf_char_ngrams"]
                for src_k, dst_k in _hybrid_tfidf_key_map:
                    if src_k in config:
                        tf_cfg[dst_k] = config[src_k]
                if len(tf_terms) > DEFAULT_MAX_FEATURES:
                    tf_cfg["max_features"] = len(tf_terms)
                if "tfidf_idf" in config:
                    tf_cfg["idf"] = config["tfidf_idf"]
                    tf_proj = TfidfProjector.from_config(tf_cfg)
                    if tfidf_idf_array is not None:
                        bin_idf = np.asarray(tfidf_idf_array, dtype=np.float64)
                        if (
                            bin_idf.ndim != 1
                            or len(bin_idf) != len(tf_terms)
                            or not np.all(np.isfinite(bin_idf))
                            or not np.allclose(
                                np.asarray(tf_proj._idf, dtype=np.float64),
                                bin_idf,
                                rtol=0.0,
                                atol=1e-5,
                            )
                        ):
                            raise ValueError(
                                "Saved hybrid projector inline tfidf_idf mismatches binary IDF tensor"
                            )
                    proj._tfidf_projector = tf_proj
                else:
                    if tfidf_idf_array is None:
                        raise ValueError("Saved hybrid projector missing TF-IDF idf weights")
                    proj._tfidf_projector = TfidfProjector.from_config(
                        tf_cfg, idf_array=tfidf_idf_array
                    )
            raw_weights = config.get("spectral_weights", ())
            if not isinstance(raw_weights, (list, tuple)):
                raise ValueError("Saved hybrid projector spectral weights must be a sequence")
            weights = tuple(float(w) for w in raw_weights)
            if len(terms) != len(weights) or any(not math.isfinite(w) for w in weights):
                raise ValueError("Saved hybrid projector spectral terms and weights length mismatch")
            with np.errstate(over="ignore", invalid="ignore"):
                weights_arr = (
                    np.asarray(weights, dtype=np.float32)
                    if weights
                    else np.empty(0, dtype=np.float32)
                )
            if not bool(np.all(np.isfinite(weights_arr))):
                raise ValueError("Saved hybrid projector spectral weights contain non-finite values")
            proj._spectral_terms = terms
            proj._spectral_vocab = {tm: i for i, tm in enumerate(terms)}
            proj._spectral_weights = weights
            proj._spectral_weights_arr = weights_arr

            raw_spec = spectral_vectors if spectral_vectors is not None else config.get("spectral_vectors")
            if raw_spec is not None and len(raw_spec) > 0:
                spec_arr = np.asarray(raw_spec, dtype=np.float32)
                if proj._direct_spectral:
                    max_r = proj._direct_spectral_dim if proj._direct_spectral_dim > 0 else spec_arr.shape[1]
                    if (
                        spec_arr.ndim != 2
                        or spec_arr.shape[0] != len(terms)
                        or not (1 <= spec_arr.shape[1] <= 1024)
                        or (proj._direct_spectral_dim > 0 and spec_arr.shape[1] != max_r)
                        or not np.all(np.isfinite(spec_arr))
                    ):
                        raise ValueError("Saved hybrid projector direct spectral_vectors have invalid shape")
                    proj._direct_spectral_dim = int(spec_arr.shape[1])
                    proj._spectral_vectors = spec_arr
                    proj._spectral_basis = None
                else:
                    if (
                        spec_arr.ndim != 2
                        or spec_arr.shape[0] != len(terms)
                        or not (1 <= spec_arr.shape[1] <= dense)
                        or not np.all(np.isfinite(spec_arr))
                    ):
                        raise ValueError("Saved hybrid projector spectral_vectors have invalid shape")
                    proj._spectral_vectors = spec_arr
                    vocab_dense = proj._dense_engine.encode_batch(list(terms)).astype(np.float32)
                    aligned_basis = (spec_arr.T @ vocab_dense).astype(np.float32)
                    b_norms = np.linalg.norm(aligned_basis, axis=1, keepdims=True)
                    if np.all(b_norms.flatten() > 1e-6):
                        proj._spectral_basis = (aligned_basis / b_norms).astype(np.float32)
                    else:
                        proj._spectral_basis = proj._build_spectral_basis(spec_arr.shape[1])

            proj._update_dimension()
            raw_sub = subspace_components if subspace_components is not None else config.get("subspace_components")
            if raw_sub is not None and len(raw_sub) > 0:
                sub_arr = np.asarray(raw_sub, dtype=np.float32)
                base_dim = proj._base_dimension
                if (
                    sub_arr.ndim != 2
                    or not (1 <= sub_arr.shape[0] <= base_dim)
                    or sub_arr.shape[1] != base_dim
                    or not np.all(np.isfinite(sub_arr))
                ):
                    raise ValueError("Saved hybrid projector subspace_components have invalid shape")
                proj._subspace_components = sub_arr
            sub_w = float(config.get("subspace_weight", 0.20))
            if not math.isfinite(sub_w):
                raise ValueError("Saved hybrid projector subspace_weight must be finite")
            proj._subspace_weight = sub_w

            raw_rbf = rbf_landmarks if rbf_landmarks is not None else config.get("rbf_landmarks")
            if raw_rbf is not None and len(raw_rbf) > 0:
                rbf_arr = np.asarray(raw_rbf, dtype=np.float32)
                base_dim = proj._base_dimension
                if rbf_arr.ndim != 2 or rbf_arr.shape[0] < 1 or rbf_arr.shape[1] != base_dim or not np.all(np.isfinite(rbf_arr)):
                    raise ValueError("Saved hybrid projector rbf_landmarks have invalid shape")
                proj._rbf_landmarks = rbf_arr
                proj._rbf_gamma = float(config.get("rbf_gamma", 3.0))
                proj._rbf_weight = float(config.get("rbf_weight", 0.14))

            proj._update_dimension()
            proj._fitted = True
        elif has_any_tfidf_cfg or tfidf_idf_array is not None:
            raise ValueError("Unfitted saved hybrid projector cannot carry TF-IDF state")
        proj._digest = None
        return proj

    def projector_digest(self) -> str:
        if self._digest is None:
            self._digest = hashlib.sha256(json.dumps(self.to_config(), sort_keys=True).encode("utf-8")).hexdigest()
        return self._digest

    to_dict = to_config
    from_dict = from_config


HybridSemanticProjector = HybridProjector
UnifiedHybridProjector = HybridProjector


_BM25_WORD_SPLIT_RE = re.compile(r"[a-z0-9]+")
_BM25_CRIT_SPLIT_RE = re.compile(r"[;.]+")
_BM25_ITEM_SPLIT_RE = re.compile(r"[;]+")
_BM25_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "into",
    "or", "and", "that", "this", "these", "those", "it", "its", "my", "your",
    "our", "their", "we", "you", "i", "me", "us", "they", "them", "do", "does",
    "did", "can", "could", "will", "would", "should", "may", "might", "please",
    "how", "what", "when", "where", "why", "which", "who", "whom", "about",
    "any", "all", "if", "so", "than", "then", "also", "just", "only", "already",
    "needs", "need", "related", "issues", "problems", "requests", "questions",
    "handling", "select", "classify", "appropriate", "category", "option",
})


def _extract_bm25_feats(
    text: str,
    *,
    include_char: bool = True,
    include_prag: bool = True,
) -> Dict[str, float]:
    from system1.core.text import _stem_word as _text_stem_word, extract_pragmatic_features

    feats: Dict[str, float] = {}
    words = [w for w in _BM25_WORD_SPLIT_RE.findall(text.lower()) if len(w) >= 2]
    prev_st: Optional[str] = None
    for w in words:
        if w in _BM25_STOPWORDS:
            prev_st = None
            continue
        wk = f"w_{w}"
        feats[wk] = feats.get(wk, 0.0) + 1.0
        st = _text_stem_word(w)
        if len(st) >= 3 and st not in _BM25_STOPWORDS:
            sk = f"s_{st}"
            feats[sk] = feats.get(sk, 0.0) + 0.90
            if prev_st is not None:
                bk = f"bg_{prev_st}_{st}"
                feats[bk] = feats.get(bk, 0.0) + 1.15
            prev_st = st
        else:
            prev_st = None
        if include_char and len(w) >= 4:
            padded = f"<{w}>"
            for i in range(len(padded) - 3):
                ck = f"c4_{padded[i : i + 4]}"
                feats[ck] = feats.get(ck, 0.0) + 0.16

    if include_prag and ("{" in text or "=" in text or any(c.isdigit() for c in text)):
        raw_prag = extract_pragmatic_features(
            text,
            enable_negation=False,
            enable_num_http=True,
            enable_json_kv=True,
        )
        for term, val in raw_prag.items():
            if term.startswith(("HTTP_", "KV_")):
                feats[term] = feats.get(term, 0.0) + float(val)
    return feats


def _split_criterion_units(opt: str, desc_val: Any) -> List[str]:
    units: List[str] = []
    clean_opt = opt.replace("_", " ").replace("-", " ")
    if isinstance(desc_val, (list, tuple)):
        for item in desc_val:
            for sub in _BM25_ITEM_SPLIT_RE.split(str(item)):
                if sub.strip():
                    units.append(sub.strip())
    elif isinstance(desc_val, str) and desc_val.strip():
        for sub in _BM25_CRIT_SPLIT_RE.split(desc_val):
            if sub.strip():
                units.append(sub.strip())
    return units or [clean_opt]


def _solve_dual_or_primal_ridge_np(
    X: np.ndarray,
    Y: np.ndarray,
    reg: float,
    sample_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Closed-form weighted multi-output ridge regression in pure NumPy."""
    n, d = X.shape
    if sample_weights is None:
        w = np.full(n, 1.0 / max(1, n), dtype=np.float64)
    else:
        w_raw = np.maximum(1e-6, np.asarray(sample_weights, dtype=np.float64))
        w = w_raw / np.sum(w_raw)
    mu_x = (w[:, None] * X).sum(axis=0)
    mu_y = (w[:, None] * Y).sum(axis=0)
    Xc = X - mu_x
    Yc = Y - mu_y
    sw = np.sqrt(w * float(n))
    Xw = Xc * sw[:, None]
    Yw = Yc * sw[:, None]
    if n < d:
        kmat = Xw @ Xw.T
        kmat[np.diag_indices(n)] += float(reg)
        dual = np.linalg.solve(kmat, Yw)
        U = (Xw.T @ dual).T
    else:
        xtx = Xw.T @ Xw
        xtx[np.diag_indices(d)] += float(reg)
        U = np.linalg.solve(xtx, Xw.T @ Yw).T
    c = mu_y - U @ mu_x
    return U, c


@dataclass(frozen=True)
class RouterDecisionResult:
    """Lightweight isolated zero/few-shot decision output matching DecisionResult attributes."""

    schema_name: str
    schema_digest: str
    prompt: str
    values: Dict[str, Any]
    confidences: Dict[str, float]
    probabilities: Dict[str, Dict[str, float]]
    conformal_sets: Dict[str, Tuple[Any, ...]]
    is_ambiguous: bool
    latency_ms: float
    alpha: float = 0.05
    receipt: Optional[Any] = None


class SmartsCrossZeroFewRouter:
    """Zero-pretrained-weight 0/few-shot router combining BM25F, MaxSim criteria, concept interaction, and ridge."""

    def __init__(
        self,
        schema_cls: Any,
        field_name: Optional[str] = None,
        options: Optional[Sequence[str]] = None,
        descriptions: Optional[Mapping[str, Any]] = None,
        instruction: str = "",
        *,
        disable_concept_gate: bool = False,
        dimension: int = 384,
    ) -> None:
        from system1.core.model import SystemOneModel
        from system1.core.schema import BooleanField, DecisionSchema

        if isinstance(schema_cls, DecisionSchema):
            s_inst = schema_cls
            self.schema_name = s_inst.schema_name
        elif isinstance(schema_cls, type) and issubclass(schema_cls, DecisionSchema):
            s_inst = schema_cls()
            self.schema_name = schema_cls.__name__
        else:
            raise TypeError(f"Expected DecisionSchema subclass or instance, got {type(schema_cls).__name__}")

        if field_name is None:
            field_name = next(iter(s_inst.fields))
        if field_name not in s_inst.fields:
            raise KeyError(f"Field {field_name!r} not found in schema")

        f_def = s_inst.fields[field_name]
        self.is_boolean = isinstance(f_def, BooleanField)
        if options is None:
            if self.is_boolean:
                options = ("False", "True")
            elif hasattr(f_def, "options"):
                options = list(f_def.options)
            else:
                raise ValueError(f"Field {field_name!r} does not define discrete options")
        if descriptions is None:
            if self.is_boolean:
                descriptions = {
                    "False": f_def.false_description,
                    "True": f_def.true_description,
                }
            else:
                descriptions = dict(getattr(f_def, "descriptions", {}) or {})
        if not instruction:
            instruction = str(getattr(f_def, "description", "") or "")

        self.options: List[str] = [str(o) for o in options]
        if not self.options:
            raise ValueError("options must not be empty")
        self.K = len(self.options)
        self.opt_to_idx: Dict[str, int] = {o: i for i, o in enumerate(self.options)}
        self.descriptions = dict(descriptions)
        self.instruction = instruction
        self.field_name = field_name
        self.disable_concept_gate = bool(disable_concept_gate)
        self.schema_digest = s_inst.schema_digest()
        self.rich_descriptions = (
            float(np.mean([len(str(self.descriptions.get(o, "")).split()) for o in self.options])) >= 4.0
        )
        self.s1_model = SystemOneModel(s_inst, dimension=dimension, backend="numpy")
        self._compile_bm25f_and_concepts()
        self.fewshot_active = False
        self.k_per_class = 0
        self.few_proj: Optional[Any] = None
        self.few_vocab: Dict[str, int] = {}
        self.few_W: Optional[np.ndarray] = None
        self.few_b: Optional[np.ndarray] = None
        self.few_protos: Optional[np.ndarray] = None
        self.few_X: Optional[np.ndarray] = None
        self.few_y: Optional[np.ndarray] = None
        self.few_idf: Optional[np.ndarray] = None
        self._few_examples: List[Tuple[str, str]] = []
        self._calibrated = False
        self._conformal_q = 1.0
        self._cal_alpha = 0.10
        self._cal_scores: Optional[np.ndarray] = None

    def _normalize_label_str(self, label: Any) -> str:
        if self.is_boolean and isinstance(label, bool):
            key = "True" if label else "False"
        else:
            key = str(label)
        if key not in self.opt_to_idx:
            raise ValueError(f"Unknown label {label!r}; expected one of {self.options}")
        return key

    def _public_value(self, label_str: str) -> Any:
        if self.is_boolean:
            if label_str == "True":
                return True
            if label_str == "False":
                return False
        return label_str

    def _compile_bm25f_and_concepts(self) -> None:
        K = self.K
        name_feats: List[Dict[str, float]] = []
        pool_feats: List[Dict[str, float]] = []
        crit_units: List[Tuple[int, Dict[str, float], Dict[str, float]]] = []
        opt_concept_pool: List[Counter[str]] = [Counter() for _ in range(K)]

        for c_idx, opt in enumerate(self.options):
            clean_opt = opt.replace("_", " ").replace("-", " ")
            nf = _extract_bm25_feats(clean_opt, include_char=True)
            name_feats.append(nf)
            units = _split_criterion_units(opt, self.descriptions.get(opt, ""))
            pf: Counter[str] = Counter()
            for k, v in nf.items():
                pf[k] += v * 1.5
            for ck, cv in match_semantic_concepts(clean_opt).items():
                opt_concept_pool[c_idx][ck] += float(cv) * 1.2
            if len(units) == 1:
                uf = _extract_bm25_feats(f"{clean_opt} {units[0]}", include_char=True)
                uc = dict(match_semantic_concepts(f"{clean_opt}: {units[0]}"))
                crit_units.append((c_idx, uf, uc))
                for k, v in uf.items():
                    pf[k] += v
                for ck, cv in uc.items():
                    opt_concept_pool[c_idx][ck] += float(cv)
            else:
                for u in units:
                    uf = _extract_bm25_feats(u, include_char=True)
                    uc = dict(match_semantic_concepts(f"{clean_opt}: {u}"))
                    crit_units.append((c_idx, uf, uc))
                    for k, v in uf.items():
                        pf[k] += v
                    for ck, cv in uc.items():
                        opt_concept_pool[c_idx][ck] += float(cv)
            pool_feats.append(dict(pf))

        self.M = len(crit_units)
        self.crit_opt_idx = np.array([c for c, _, _ in crit_units], dtype=np.int32)
        self.opt_to_crit_idxs = [np.where(self.crit_opt_idx == c)[0] for c in range(K)]

        all_ck = sorted({ck for cp in opt_concept_pool for ck in cp})
        if self.disable_concept_gate:
            self.concept_gate = 1.0
        elif all_ck:
            c_mat = np.zeros((K, len(all_ck)), dtype=np.float64)
            ck_to_i = {ck: i for i, ck in enumerate(all_ck)}
            for c_idx, cp in enumerate(opt_concept_pool):
                for ck, cv in cp.items():
                    c_mat[c_idx, ck_to_i[ck]] = cv
            max_strength = float(np.mean(np.max(c_mat, axis=1)))
            norms = np.linalg.norm(c_mat, axis=1, keepdims=True)
            c_unit = c_mat / np.maximum(1e-8, norms)
            sim_mat = c_unit @ c_unit.T
            off_diag = (float(np.sum(sim_mat)) - float(np.trace(sim_mat))) / max(1, K * (K - 1))
            ortho = max(0.0, 1.0 - off_diag)
            self.concept_gate = ortho * min(1.0, max(0.0, (max_strength - 2.5) / 8.0))
        else:
            self.concept_gate = 0.0

        df_opt: Counter[str] = Counter()
        for pf in pool_feats:
            for t in pf:
                df_opt[t] += 1
        df_crit: Counter[str] = Counter()
        for _, uf, _ in crit_units:
            for t in uf:
                df_crit[t] += 1

        def _idf(df: int, n_docs: int) -> float:
            return math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        avg_name = max(1.0, float(np.mean([sum(f.values()) for f in name_feats])))
        avg_pool = max(1.0, float(np.mean([sum(f.values()) for f in pool_feats])))
        avg_crit = max(1.0, float(np.mean([sum(f.values()) for _, f, _ in crit_units])))
        k1, b_field = 1.2, 0.45

        self.opt_lex_tbl: Dict[str, np.ndarray] = {}
        self.crit_lex_tbl: Dict[str, np.ndarray] = {}
        for t in set(df_opt.keys()) | set(df_crit.keys()):
            idf = 0.65 * _idf(df_opt.get(t, 0), K) + 0.35 * _idf(df_crit.get(t, 0), max(1, self.M))
            v_opt = np.zeros(K, dtype=np.float64)
            for c_idx in range(K):
                tf_n = name_feats[c_idx].get(t, 0.0)
                if tf_n > 0.0:
                    norm_n = (1.0 - b_field) + b_field * (sum(name_feats[c_idx].values()) / avg_name)
                    v_opt[c_idx] += 2.3 * (tf_n * (k1 + 1.0)) / (tf_n + k1 * norm_n)
                tf_p = pool_feats[c_idx].get(t, 0.0)
                if tf_p > 0.0:
                    norm_p = (1.0 - b_field) + b_field * (sum(pool_feats[c_idx].values()) / avg_pool)
                    v_opt[c_idx] += 1.2 * (tf_p * (k1 + 1.0)) / (tf_p + k1 * norm_p)
            if np.any(v_opt > 0.0):
                self.opt_lex_tbl[t] = idf * v_opt

            v_crit = np.zeros(self.M, dtype=np.float64)
            any_c = False
            for m_idx, (_, uf, _) in enumerate(crit_units):
                tf_u = uf.get(t, 0.0)
                if tf_u > 0.0:
                    any_c = True
                    norm_u = (1.0 - 0.50) + 0.50 * (sum(uf.values()) / avg_crit)
                    v_crit[m_idx] = idf * (tf_u * (k1 + 1.0)) / (tf_u + k1 * norm_u)
            if any_c:
                self.crit_lex_tbl[t] = v_crit

        opt_sq = np.zeros(K, dtype=np.float64)
        for v in self.opt_lex_tbl.values():
            opt_sq += v * v
        opt_inv = 1.0 / np.maximum(1e-6, np.sqrt(opt_sq))
        for t in self.opt_lex_tbl:
            self.opt_lex_tbl[t] *= opt_inv

        crit_sq = np.zeros(self.M, dtype=np.float64)
        for v in self.crit_lex_tbl.values():
            crit_sq += v * v
        crit_inv = 1.0 / np.maximum(1e-6, np.sqrt(crit_sq))
        for t in self.crit_lex_tbl:
            self.crit_lex_tbl[t] *= crit_inv

        df_co: Counter[str] = Counter()
        for cp in opt_concept_pool:
            for ck in cp:
                df_co[ck] += 1
        df_cc: Counter[str] = Counter()
        for _, _, uc in crit_units:
            for ck in uc:
                df_cc[ck] += 1

        self.opt_concept_tbl: Dict[str, np.ndarray] = {}
        self.crit_concept_tbl: Dict[str, np.ndarray] = {}
        for ck in set(df_co.keys()) | set(df_cc.keys()):
            idf_ck = 0.65 * _idf(df_co.get(ck, 0), K) + 0.35 * _idf(df_cc.get(ck, 0), max(1, self.M))
            vo = np.array(
                [
                    idf_ck * math.log1p(opt_concept_pool[c].get(ck, 0.0))
                    if opt_concept_pool[c].get(ck, 0.0) > 0
                    else 0.0
                    for c in range(K)
                ],
                dtype=np.float64,
            )
            if np.any(vo > 0.0):
                self.opt_concept_tbl[ck] = vo
            vc = np.zeros(self.M, dtype=np.float64)
            any_cc = False
            for m_idx, (_, _, uc) in enumerate(crit_units):
                val = uc.get(ck, 0.0)
                if val > 0.0:
                    any_cc = True
                    vc[m_idx] = idf_ck * math.log1p(val)
            if any_cc:
                self.crit_concept_tbl[ck] = vc

        oc_sq = np.zeros(K, dtype=np.float64)
        for v in self.opt_concept_tbl.values():
            oc_sq += v * v
        oc_inv = 1.0 / np.maximum(1e-6, np.sqrt(oc_sq))
        for ck in self.opt_concept_tbl:
            self.opt_concept_tbl[ck] *= oc_inv

        cc_sq = np.zeros(self.M, dtype=np.float64)
        for v in self.crit_concept_tbl.values():
            cc_sq += v * v
        cc_inv = 1.0 / np.maximum(1e-6, np.sqrt(cc_sq))
        for ck in self.crit_concept_tbl:
            self.crit_concept_tbl[ck] *= cc_inv

    def _featurize_with_concepts(self, lex_feats: Mapping[str, float], q_concepts: Mapping[str, float]) -> Dict[str, float]:
        feats = dict(lex_feats)
        if self.concept_gate > 0.15:
            for ck, cv in q_concepts.items():
                feats[f"CNC_{ck}"] = float(cv) * 1.4
        return feats

    def _vectorize_few(self, feats: Mapping[str, float]) -> np.ndarray:
        vec = np.zeros(len(self.few_vocab), dtype=np.float64)
        vocab = self.few_vocab
        idf = self.few_idf
        for t, w in feats.items():
            idx = vocab.get(t)
            if idx is not None and idf is not None:
                vec[idx] = math.log1p(w) * idf[idx]
        nrm = float(np.linalg.norm(vec))
        if nrm > 1e-12:
            vec /= nrm
        return vec

    def fit_few_shot(
        self,
        examples_or_prompts: Union[Sequence[Tuple[str, Any]], Sequence[str]],
        labels: Optional[Sequence[Any]] = None,
    ) -> "SmartsCrossZeroFewRouter":
        from system1.core.text import TfidfProjector

        if labels is not None:
            if len(examples_or_prompts) != len(labels):
                raise ValueError("prompts and labels must have equal length")
            pairs = [(str(p), self._normalize_label_str(lbl)) for p, lbl in zip(examples_or_prompts, labels)]
        else:
            pairs = [(str(item[0]), self._normalize_label_str(item[1])) for item in examples_or_prompts]  # type: ignore[index]

        self._few_examples = list(pairs)
        self._calibrated = False
        self._cal_scores = None
        if not pairs:
            self.fewshot_active = False
            self.few_proj = None
            self.k_per_class = 0
            return self

        all_texts = [txt for txt, _ in pairs]
        all_labels_str = [lbl for _, lbl in pairs]
        all_labels = [self.opt_to_idx[lbl] for lbl in all_labels_str]
        counts = np.bincount(all_labels, minlength=self.K)
        self.k_per_class = max(1, int(np.min(counts[counts > 0]))) if np.any(counts > 0) else 0
        schema_texts = [
            f"{opt.replace('_', ' ')}: {' '.join(_split_criterion_units(opt, self.descriptions.get(opt, '')))}"
            for opt in self.options
        ]

        if self.rich_descriptions:
            self.few_proj = None
            shot_fmaps = [
                self._featurize_with_concepts(_extract_bm25_feats(t, include_char=True), match_semantic_concepts(t))
                for t in all_texts
            ]
            schema_fmaps = [
                self._featurize_with_concepts(_extract_bm25_feats(s, include_char=True), match_semantic_concepts(s))
                for s in schema_texts
            ]
            df: Counter[str] = Counter()
            for fm in shot_fmaps + schema_fmaps:
                for t in fm:
                    df[t] += 1
            n_docs = len(shot_fmaps) + len(schema_fmaps)
            terms_sorted = sorted(df.keys(), key=lambda t: (-df[t], t))[:1536]
            self.few_vocab = {t: i for i, t in enumerate(terms_sorted)}
            self.few_idf = np.array([math.log((n_docs + 1.0) / (df[t] + 1.0)) + 1.0 for t in terms_sorted], dtype=np.float64)
            X_shots = np.vstack([self._vectorize_few(fm) for fm in shot_fmaps])
            X_schema = np.vstack([self._vectorize_few(fm) for fm in schema_fmaps])
            reg = 0.85 if self.k_per_class <= 1 else (0.30 if self.k_per_class <= 3 else 0.16)
        else:
            self.few_vocab = {}
            self.few_idf = None
            self.few_proj = TfidfProjector.fit(
                all_texts + schema_texts,
                max_features=2048,
                labels=all_labels_str + list(self.options),
                selection="chi2",
                char_ngrams=(3, 5),
            )
            X_shots = np.asarray(self.few_proj.project_batch(all_texts), dtype=np.float64)
            X_schema = np.asarray(self.few_proj.project_batch(schema_texts), dtype=np.float64)
            reg = 0.45 if self.k_per_class <= 1 else (0.20 if self.k_per_class <= 3 else 0.10)

        w_prior = math.sqrt(max(0.25, 1.5 / (1.0 + 0.4 * float(self.k_per_class))))
        X_aug = np.vstack([X_shots, w_prior * X_schema])
        Y_shots = np.zeros((len(all_labels), self.K), dtype=np.float64)
        for i, c in enumerate(all_labels):
            Y_shots[i, c] = 1.0
        Y_aug = np.vstack([Y_shots, w_prior * np.eye(self.K, dtype=np.float64)])

        self.few_W, self.few_b = _solve_dual_or_primal_ridge_np(X_aug, Y_aug, reg=reg)
        self.few_X = X_shots
        self.few_y = np.array(all_labels, dtype=np.int32)
        self.few_protos = np.zeros((self.K, X_shots.shape[1]), dtype=np.float64)
        for c in range(self.K):
            mask = self.few_y == c
            p = np.mean(X_shots[mask], axis=0) if np.any(mask) else X_schema[c]
            nrm = float(np.linalg.norm(p))
            if nrm > 1e-12:
                self.few_protos[c] = p / nrm
        self.fewshot_active = True
        return self

    def learn(self, prompt: str, label: Any) -> "SmartsCrossZeroFewRouter":
        """Incrementally add one labeled example and refresh few-shot weights."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        norm_lbl = self._normalize_label_str(label)
        self._few_examples.append((prompt, norm_lbl))
        return self.fit_few_shot(self._few_examples)

    def calibrate(
        self,
        prompts: Sequence[str],
        labels: Sequence[Any],
        *,
        alpha: float = 0.10,
    ) -> float:
        """Fit split-conformal LAC quantile over held-out calibration examples."""
        if len(prompts) != len(labels):
            raise ValueError("prompts and labels must have equal length")
        if not (0.0 < float(alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        n = len(prompts)
        if n == 0:
            self._calibrated = False
            self._conformal_q = 1.0
            self._cal_alpha = float(alpha)
            self._cal_scores = None
            return self._conformal_q
        scores = np.empty(n, dtype=np.float64)
        for i, (prompt, raw_lbl) in enumerate(zip(prompts, labels)):
            _, probs = self.score_and_predict(str(prompt))
            lbl_idx = self.opt_to_idx[self._normalize_label_str(raw_lbl)]
            scores[i] = 1.0 - float(probs[lbl_idx])
        sorted_scores = np.sort(scores)
        rank = int(math.ceil((n + 1) * (1.0 - float(alpha))))
        self._conformal_q = 1.0 if rank > n else float(sorted_scores[max(0, rank - 1)])
        self._cal_alpha = float(alpha)
        self._cal_scores = sorted_scores
        self._calibrated = True
        return self._conformal_q

    def score_and_predict(self, text: str) -> Tuple[Any, np.ndarray]:
        if not isinstance(text, str):
            raise TypeError(f"Prompt must be a string, got {type(text).__name__}")
        lf = _extract_bm25_feats(text, include_char=True)
        opt_l = np.zeros(self.K, dtype=np.float64)
        crit_l = np.zeros(self.M, dtype=np.float64)
        ql_sq = 0.0
        opt_tbl, crit_tbl = self.opt_lex_tbl, self.crit_lex_tbl
        for t, tf in lf.items():
            qw = math.log1p(tf)
            vo = opt_tbl.get(t)
            if vo is not None:
                opt_l += qw * vo
            vc = crit_tbl.get(t)
            if vc is not None:
                crit_l += qw * vc
                ql_sq += qw * qw
        if ql_sq > 1e-12:
            inv_q = 1.0 / math.sqrt(ql_sq)
            opt_l *= inv_q
            crit_l *= inv_q
        crit_l_max = np.array(
            [float(np.max(crit_l[idxs])) if len(idxs) else 0.0 for idxs in self.opt_to_crit_idxs],
            dtype=np.float64,
        )
        s_lex_c = 0.60 * opt_l + 0.40 * crit_l_max
        s_lex_c -= float(np.mean(s_lex_c))

        qc: Dict[str, float] = {}
        if self.concept_gate > 0.05:
            qc = match_semantic_concepts(text)
            opt_c = np.zeros(self.K, dtype=np.float64)
            crit_c = np.zeros(self.M, dtype=np.float64)
            qc_sq = 0.0
            for ck, cv in qc.items():
                qw = math.log1p(cv)
                vo = self.opt_concept_tbl.get(ck)
                if vo is not None:
                    opt_c += qw * vo
                vc = self.crit_concept_tbl.get(ck)
                if vc is not None:
                    crit_c += qw * vc
                    qc_sq += qw * qw
            if qc_sq > 1e-12:
                inv_c = 1.0 / math.sqrt(qc_sq)
                opt_c *= inv_c
                crit_c *= inv_c
            crit_c_max = np.array(
                [float(np.max(crit_c[idxs])) if len(idxs) else 0.0 for idxs in self.opt_to_crit_idxs],
                dtype=np.float64,
            )
            s_cnc_c = 0.40 * opt_c + 0.60 * crit_c_max
            s_cnc_c -= float(np.mean(s_cnc_c))
        else:
            s_cnc_c = np.zeros(self.K, dtype=np.float64)

        s1_raw = np.asarray(self.s1_model.forward_single(text).fields[self.field_name].logits, dtype=np.float64).ravel()
        s1_c = s1_raw - float(np.mean(s1_raw))
        s1_w = 0.42 if self.concept_gate > 0.90 else 0.28
        zs = 1.50 * s_lex_c + (1.10 * self.concept_gate) * s_cnc_c + s1_w * s1_c

        if not self.fewshot_active or self.few_W is None or self.few_b is None:
            probs = np.exp((zs - np.max(zs)) * 4.0)
            probs /= probs.sum()
            return self._public_value(self.options[int(np.argmax(zs))]), probs

        assert self.few_protos is not None and self.few_X is not None and self.few_y is not None
        if self.few_proj is not None:
            v = np.asarray(self.few_proj.project(text), dtype=np.float64)
            rc = v @ self.few_W.T + self.few_b
            rc -= float(np.mean(rc))
            pc = self.few_protos @ v
            pc -= float(np.mean(pc))
            shot_cos = self.few_X @ v
            ec = np.array(
                [float(np.max(shot_cos[self.few_y == c])) if np.any(self.few_y == c) else 0.0 for c in range(self.K)],
                dtype=np.float64,
            )
            ec -= float(np.mean(ec))
            fs_comb = 0.72 * rc + 0.18 * pc + 0.10 * ec
            w_zs = 1.2 / (1.2 + float(self.k_per_class))
        else:
            v = self._vectorize_few(self._featurize_with_concepts(lf, qc))
            r_c = v @ self.few_W.T + self.few_b
            r_c -= float(np.mean(r_c))
            p_c = self.few_protos @ v
            p_c -= float(np.mean(p_c))
            shot_cos = self.few_X @ v
            e_c = np.array(
                [float(np.max(shot_cos[self.few_y == c])) if np.any(self.few_y == c) else 0.0 for c in range(self.K)],
                dtype=np.float64,
            )
            e_c -= float(np.mean(e_c))
            fs_comb = 0.70 * r_c + 0.20 * p_c + 0.10 * e_c
            w_zs = 2.4 / (2.4 + float(self.k_per_class))
        zs_std = float(np.std(zs))
        fs_std = float(np.std(fs_comb))
        zs_scaled = zs * (fs_std / zs_std) if (zs_std > 1e-6 and fs_std > 1e-6) else zs
        comb = w_zs * zs_scaled + (1.0 - w_zs) * fs_comb
        probs = np.exp((comb - np.max(comb)) * 4.0)
        probs /= probs.sum()
        return self._public_value(self.options[int(np.argmax(comb))]), probs

    def predict(self, prompt: str) -> Any:
        pred, _ = self.score_and_predict(prompt)
        return pred

    def predict_proba(self, prompt: str) -> Dict[str, float]:
        _, probs = self.score_and_predict(prompt)
        return {self.options[j]: float(probs[j]) for j in range(self.K)}

    def predict_batch(self, prompts: Sequence[str]) -> List[Any]:
        return [self.predict(p) for p in prompts]

    def decide(self, prompt: str, alpha: Optional[float] = None) -> RouterDecisionResult:
        eff_alpha = self._cal_alpha if alpha is None else float(alpha)
        if not (0.0 < eff_alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {eff_alpha}")
        t0 = time.perf_counter()
        pred, probs = self.score_and_predict(prompt)
        bi = self.opt_to_idx[self._normalize_label_str(pred)]
        if self._calibrated:
            if self._cal_scores is not None and len(self._cal_scores) > 0:
                n_cal = len(self._cal_scores)
                rank = int(math.ceil((n_cal + 1) * (1.0 - eff_alpha)))
                q_eff = 1.0 if rank > n_cal else float(self._cal_scores[max(0, rank - 1)])
            else:
                q_eff = self._conformal_q
            cset = tuple(
                self._public_value(self.options[j])
                for j in range(self.K)
                if (1.0 - float(probs[j])) <= q_eff + 1e-12
            )
            is_ambig = len(cset) != 1
        else:
            cset = (pred,)
            is_ambig = False
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return RouterDecisionResult(
            schema_name=self.schema_name,
            schema_digest=self.schema_digest,
            prompt=prompt,
            values={self.field_name: pred},
            confidences={self.field_name: float(probs[bi])},
            probabilities={self.field_name: {self.options[j]: float(probs[j]) for j in range(self.K)}},
            conformal_sets={self.field_name: cset},
            is_ambiguous=is_ambig,
            latency_ms=lat_ms,
            alpha=eff_alpha,
            receipt=None,
        )


__all__ = [
    "SubwordSemanticEmbeddings",
    "StaticEmbeddingTable",
    "HybridProjector",
    "HybridSemanticProjector",
    "UnifiedHybridProjector",
    "SmartsCrossZeroFewRouter",
    "RouterDecisionResult",
    "set_embedding_cache_maxsize",
    "clear_embedding_caches",
    "_build_default_table",
    "_stem_word",
    "_ANCHOR_STOPWORDS",
    "match_semantic_concepts",
    "compute_strict_loo_weights",
    "extract_schema_anchor_texts",
]
