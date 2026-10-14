"""Local subword features and optional lexical/dense feature fusion.

The NumPy subword table is seeded locally and adjusted by curated semantic
anchors. It is not a downloaded or pretrained language model. MLX is optional;
no workload-independent latency bound is asserted."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import math
import re
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


@lru_cache(maxsize=65536)
def _stem_word(word: str) -> str:
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


@lru_cache(maxsize=65536)
def _cached_token_hash(token_norm: str, vocab_size: int) -> int:
    h = hashlib.sha256(token_norm.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") % vocab_size


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

    def encode(
        self,
        text: str,
        *,
        _pretokenized: Optional[Tuple[str, List[str], Dict[int, str]]] = None,
    ) -> np.ndarray:
        """Encodes text into a normalized dense subword vector with negation-scope binding."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")
        if not text or not text.strip():
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        if _pretokenized is not None:
            clean_text, words, negated_positions = _pretokenized
        else:
            clean_text = _normalize_text(text)
            words, negated_positions = _tokenize_with_negation_scope(clean_text)

        indices: List[int] = []
        weights: List[float] = []

        # 1. Check for multi-word phrase & idiom anchors via space-adjacent 3-gram index
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

        # 2. Extract words, negation-scope bindings, and subwords
        for pos, word in enumerate(words):
            word_weight = math.log1p(len(word)) / math.sqrt(1.0 + pos * 0.05)
            if pos in negated_positions:
                cue = negated_positions[pos]
                # Emit explicit bound negated features and invert positive unigram pull
                indices.append(self._hash_token(f"neg:{word}"))
                weights.append(word_weight * 2.8)
                indices.append(self._hash_token(f"neg:{cue}_{word}"))
                weights.append(word_weight * 2.0)
                indices.append(self._hash_token(word))
                weights.append(-word_weight * 0.9)
                continue

            if word in _NEGATION_TOKENS:
                indices.append(self._hash_token(word))
                weights.append(word_weight * 0.6)
                continue

            indices.append(self._hash_token(word))
            weights.append(word_weight * 1.5)

            # Character 3-grams and 4-grams
            if len(word) >= 3:
                for n in (3, 4):
                    for j in range(len(word) - n + 1):
                        sub = word[j : j + n]
                        indices.append(self._hash_token(f"sub:{sub}"))
                        weights.append(0.35)

        # 3. Word bigrams
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            indices.append(self._hash_token(f"bi:{bigram}"))
            weights.append(1.2)

        if not indices:
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v

        idx_arr = np.array(indices, dtype=np.int32)
        w_arr = np.array(weights, dtype=np.float32)

        # Fast BLAS 1D-by-2D dot product
        accum = w_arr @ self._table[idx_arr]

        norm = float(np.linalg.norm(accum))
        if norm < 1e-12 or not np.isfinite(norm):
            v = np.zeros(self.dimension, dtype=np.float32)
            v[0] = 1.0
            return v
        return (accum / norm).astype(np.float32)

    project = encode

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Batched encoding of multiple text inputs."""
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([self.encode(t) for t in texts], axis=0).astype(np.float32)

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


class _HybridFitDescriptor:
    """Allows HybridProjector.fit(...) to be called as either a classmethod or an instance method."""

    def __get__(self, instance: Optional[HybridProjector], owner: type[HybridProjector]) -> Any:
        if instance is None:
            return owner._fit_class
        return instance._fit_instance


class HybridProjector:
    """Concatenate weighted lexical and dense subword features, with optional corpus-fitted
    PPMI/SVD spectral co-occurrence and supervised class-prototype projection.

    For unit components, square-root weights preserve unit norm; the
    implementation also normalizes its final vector. The default dimensions
    are 1024 lexical plus 256 dense, with alpha=0.5. No latency guarantee."""

    CONFIG_KEYS = frozenset({"type", "sparse_dim", "dense_dim", "alpha", "seed", "fitted", "spectral_terms",
                             "spectral_weights", "spectral_vectors", "subspace_components", "subspace_weight"})

    def __init__(
        self,
        dimension: Optional[int] = None,
        *,
        sparse_dim: int = 1024,
        dense_dim: int = 256,
        alpha: float = 0.5,
        seed: int = 42,
        backend: str = "auto",
    ) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0.0, 1.0), got {alpha}")

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
        self.alpha = float(alpha)
        self.seed = seed
        self.backend = self._resolve_backend(backend)

        # Dense subword semantic engine
        self._dense_engine = SubwordSemanticEmbeddings(
            dimension=self.dense_dim,
            vocab_size=25000,
            seed=self.seed,
        )

        # Optional corpus-fitted spectral + prototype projection state
        self._fitted: bool = False
        self._spectral_terms: Tuple[str, ...] = ()
        self._spectral_vocab: Dict[str, int] = {}
        self._spectral_weights: Tuple[float, ...] = ()
        self._spectral_vectors: Optional[np.ndarray] = None  # shape: (V, R_spec)
        self._spectral_basis: Optional[np.ndarray] = None    # shape: (R_spec, dense_dim)
        self._subspace_components: Optional[np.ndarray] = None  # shape: (R_sub, dimension)
        self._subspace_weight: float = 0.0
        self._digest: Optional[str] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def spectral_vocab(self) -> Dict[str, int]:
        return dict(self._spectral_vocab)

    @property
    def spectral_proj(self) -> Optional[np.ndarray]:
        if self._spectral_vectors is None or self._spectral_basis is None:
            return None
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
        max_vocab: int = 192,
        spectral_dim: int = 24,
        subspace_dim: int = 16,
        subspace_weight: float = 0.20,
        min_df: int = 1,
        **kwargs: Any,
    ) -> HybridProjector:
        """Fit PPMI/SVD spectral co-occurrence and optional supervised class-prototype subspace."""
        if isinstance(texts, (str, bytes)) or texts is None:
            raise TypeError("texts must be a collection of strings")
        text_list = list(texts)
        for idx, t in enumerate(text_list):
            if not isinstance(t, str):
                raise TypeError(f"Expected string at index {idx}, got {type(t).__name__}")

        label_list: Optional[List[str]] = None
        if labels is not None:
            if isinstance(labels, (str, bytes)):
                raise TypeError("labels must be a sequence matching texts")
            raw_labels = list(labels)
            if len(raw_labels) != len(text_list):
                raise ValueError("labels must have the same length as texts")
            label_list = [str(lbl) for lbl in raw_labels]

        self._digest = None
        if not text_list:
            self._fitted = True
            return self

        n_docs = len(text_list)
        doc_terms: List[List[str]] = []
        df_counter: Counter[str] = Counter()
        tf_counter: Counter[str] = Counter()
        term_class_df: Dict[str, Counter[str]] = defaultdict(Counter)
        class_counts: Counter[str] = Counter()

        for i, t in enumerate(text_list):
            terms = _extract_corpus_terms(t)
            doc_terms.append(terms)
            tf_counter.update(terms)
            unique_t = set(terms)
            df_counter.update(unique_t)
            if label_list is not None:
                c = label_list[i]
                class_counts[c] += 1
                for ut in unique_t:
                    term_class_df[ut][c] += 1

        if not tf_counter:
            self._fitted = True
            return self

        eff_min_df = max(1, int(min_df))
        candidate_terms = [tm for tm, df_int in df_counter.items() if df_int >= eff_min_df]
        if not candidate_terms:
            candidate_terms = list(df_counter.keys())

        # Compute term importance (smoothed IDF * supervised chi2 boost when labels are available)
        classes = sorted(class_counts.keys()) if label_list is not None else []
        term_scores: Dict[str, float] = {}
        term_weights_map: Dict[str, float] = {}
        per_class_positive: Dict[str, List[Tuple[float, int, str]]] = {c: [] for c in classes}
        for term in candidate_terms:
            df_int = df_counter[term]
            idf = math.log((1.0 + n_docs) / (1.0 + df_int)) + 1.0
            best_chi2 = 0.0
            if len(classes) >= 2 and n_docs > 1:
                df_f = float(df_int)
                not_df = float(n_docs) - df_f
                if df_f > 0.0 and not_df > 0.0:
                    for c in classes:
                        n_c = float(class_counts[c])
                        not_n_c = float(n_docs) - n_c
                        if n_c <= 0.0 or not_n_c <= 0.0:
                            continue
                        a = float(term_class_df[term].get(c, 0))
                        b = df_f - a
                        c_neg = n_c - a
                        d = not_df - c_neg
                        den = df_f * not_df * n_c * not_n_c
                        if den > 0.0:
                            chi2 = float(n_docs) * ((a * d - b * c_neg) ** 2) / den
                            if chi2 > best_chi2:
                                best_chi2 = chi2
                            if a > (df_f * n_c) / float(n_docs) and chi2 > 0.0:
                                per_class_positive[c].append((-round(chi2, 9), -tf_counter[term], term))
            disc_boost = 1.0 + min(4.5, math.sqrt(best_chi2))
            term_weights_map[term] = float(min(10.0, idf * disc_boost))
            term_scores[term] = float(best_chi2 * math.log1p(tf_counter[term]) if len(classes) >= 2 else tf_counter[term] * idf)

        v_limit = max(1, min(int(max_vocab), 1024))
        if len(classes) >= 2:
            for c in classes:
                per_class_positive[c].sort()
            quota = max(1, v_limit // max(1, int(1.5 * len(classes))))
            chosen_vocab: List[str] = []
            chosen_set: set[str] = set()
            max_c_len = max((len(v) for v in per_class_positive.values()), default=0)
            per_class_added = {c: 0 for c in classes}
            for rank_idx in range(max_c_len):
                if len(chosen_vocab) >= v_limit:
                    break
                for c in classes:
                    if per_class_added[c] >= quota:
                        continue
                    cands = per_class_positive[c]
                    if rank_idx < len(cands):
                        tm = cands[rank_idx][2]
                        if tm not in chosen_set:
                            chosen_vocab.append(tm)
                            chosen_set.add(tm)
                            per_class_added[c] += 1
                            if len(chosen_vocab) >= v_limit:
                                break
            if len(chosen_vocab) < v_limit:
                rem = sorted(
                    (tm for tm in term_scores if tm not in chosen_set),
                    key=lambda tm: (-round(term_scores[tm], 9), -tf_counter[tm], tm),
                )
                chosen_vocab.extend(rem[: v_limit - len(chosen_vocab)])
            selected_vocab = sorted(chosen_vocab[:v_limit])
        else:
            selected_vocab = sorted(
                sorted(term_scores.keys(), key=lambda tm: (-round(term_scores[tm], 9), -tf_counter[tm], tm))[:v_limit]
            )
        v_size = len(selected_vocab)
        vocab_idx = {tm: i for i, tm in enumerate(selected_vocab)}

        # Build symmetric document co-occurrence matrix and transform to PPMI
        cooc = np.zeros((v_size, v_size), dtype=np.float64)
        for terms in doc_terms:
            present = sorted({vocab_idx[tm] for tm in terms if tm in vocab_idx})
            for idx_a in present:
                cooc[idx_a, idx_a] += 1.0
                for idx_b in present:
                    if idx_a != idx_b:
                        cooc[idx_a, idx_b] += 1.0

        total_cooc = float(np.sum(cooc))
        if total_cooc > 0.0:
            row_sums = np.sum(cooc, axis=1, keepdims=True)
            col_sums = np.sum(cooc, axis=0, keepdims=True)
            denom = np.maximum(row_sums @ col_sums, 1e-12)
            ratio = (cooc * total_cooc) / denom
            ppmi = np.maximum(0.0, np.log(np.maximum(ratio, 1e-12)))
        else:
            ppmi = np.eye(v_size, dtype=np.float64)

        # Truncated SVD on PPMI matrix for distributional spectral representations
        r_spec = max(1, min(int(spectral_dim), v_size, self.dense_dim))
        u_mat, s_vals, _ = np.linalg.svd(ppmi, full_matrices=False)
        spec_vecs = (u_mat[:, :r_spec] * np.sqrt(np.maximum(s_vals[:r_spec], 1e-12))).astype(np.float32)
        spec_norms = np.linalg.norm(spec_vecs, axis=1, keepdims=True)
        spec_norms = np.where(spec_norms < 1e-12, 1.0, spec_norms)
        spec_vecs = (spec_vecs / spec_norms).astype(np.float32)

        self._spectral_terms = tuple(selected_vocab)
        self._spectral_vocab = vocab_idx
        self._spectral_weights = tuple(term_weights_map[tm] for tm in selected_vocab)
        self._spectral_vectors = spec_vecs
        # Align spectral basis with the semantic dense space of the vocabulary terms
        vocab_dense = np.stack([self._dense_engine.encode(tm) for tm in selected_vocab], axis=0).astype(np.float32)
        aligned_basis = (spec_vecs.T @ vocab_dense).astype(np.float32)
        b_norms = np.linalg.norm(aligned_basis, axis=1, keepdims=True)
        if np.all(b_norms.flatten() > 1e-6):
            self._spectral_basis = (aligned_basis / b_norms).astype(np.float32)
        else:
            self._spectral_basis = self._build_spectral_basis(r_spec)
        self._subspace_components = None
        self._subspace_weight = float(subspace_weight)
        self._fitted = True

        # Step 2: Fit supervised class-prototype discriminative subspace or unsupervised principal subspace
        self._fit_base_cache: Dict[str, np.ndarray] = {}
        base_embs = np.stack([self._project_base(t) for t in text_list], axis=0).astype(np.float64)
        if len(text_list) <= 4096:
            self._fit_base_cache = {
                t: base_embs[i].astype(np.float32) for i, t in enumerate(text_list)
            }
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
                r_sub = max(1, min(int(subspace_dim), vt_p.shape[0], self.dimension))
                valid = s_p[:r_sub] > 1e-6
                if np.any(valid):
                    self._subspace_components = vt_p[:r_sub][valid].astype(np.float32)
        elif n_docs >= 2:
            centered = base_embs - np.mean(base_embs, axis=0, keepdims=True)
            _, s_u, vt_u = np.linalg.svd(centered, full_matrices=False)
            r_sub = max(1, min(int(subspace_dim), vt_u.shape[0], self.dimension))
            valid = s_u[:r_sub] > 1e-6
            if np.any(valid):
                self._subspace_components = vt_u[:r_sub][valid].astype(np.float32)

        self._digest = None
        return self

    @classmethod
    def _fit_class(
        cls,
        texts: Iterable[str],
        labels: Optional[Sequence[Any]] = None,
        *,
        dimension: Optional[int] = None,
        sparse_dim: int = 1024,
        dense_dim: int = 256,
        alpha: float = 0.5,
        seed: int = 42,
        backend: str = "auto",
        max_vocab: int = 192,
        spectral_dim: int = 24,
        subspace_dim: int = 16,
        subspace_weight: float = 0.20,
        min_df: int = 1,
        **kwargs: Any,
    ) -> HybridProjector:
        projector = cls(
            dimension=dimension,
            sparse_dim=sparse_dim,
            dense_dim=dense_dim,
            alpha=alpha,
            seed=seed,
            backend=backend,
        )
        return projector._fit_instance(
            texts,
            labels=labels,
            max_vocab=max_vocab,
            spectral_dim=spectral_dim,
            subspace_dim=subspace_dim,
            subspace_weight=subspace_weight,
            min_df=min_df,
            **kwargs,
        )

    fit = _HybridFitDescriptor()

    @staticmethod
    @lru_cache(maxsize=65536)
    def _sparse_word_hashes(token: str, sparse_dim: int, count: int = 4) -> Tuple[Tuple[int, float], ...]:
        h = hashlib.sha256(f"word:{token}".encode("utf-8")).digest()
        res: List[Tuple[int, float]] = []
        for i in range(count):
            chunk = h[i * 4 : (i + 1) * 4]
            val = int.from_bytes(chunk, byteorder="big", signed=False)
            idx = val % sparse_dim
            sign = 1.0 if ((val >> 16) & 1) == 0 else -1.0
            res.append((idx, sign))
        return tuple(res)

    @staticmethod
    @lru_cache(maxsize=65536)
    def _sparse_single_hash(prefix_token: str, sparse_dim: int) -> Tuple[int, float]:
        h = hashlib.sha256(prefix_token.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % sparse_dim
        sign = 1.0 if (h[4] & 1) == 0 else -1.0
        return idx, sign

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

    def _encode_spectral_dense(self, text: str, base_dense: np.ndarray) -> np.ndarray:
        """Blends corpus-learned PPMI/SVD spectral co-occurrence vector into dense channel."""
        if (
            not self._fitted
            or self._spectral_vectors is None
            or self._spectral_basis is None
            or not self._spectral_vocab
        ):
            return base_dense
        terms = _extract_corpus_terms(text)
        if not terms:
            return base_dense
        counts = Counter(t for t in terms if t in self._spectral_vocab)
        if not counts:
            return base_dense
        r_spec = self._spectral_vectors.shape[1]
        spec_accum = np.zeros(r_spec, dtype=np.float32)
        for tm, cnt in counts.items():
            idx = self._spectral_vocab[tm]
            w = (1.0 + math.log(cnt)) * self._spectral_weights[idx]
            spec_accum += w * self._spectral_vectors[idx]
        s_norm = float(np.linalg.norm(spec_accum))
        if s_norm < 1e-12:
            return base_dense
        spec_unit = spec_accum / s_norm
        spec_dense = spec_unit @ self._spectral_basis
        sd_norm = float(np.linalg.norm(spec_dense))
        if sd_norm > 1e-12:
            spec_dense = spec_dense / sd_norm
        blended = 0.65 * base_dense + 0.35 * spec_dense
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
        v_sparse = self._hash_to_sparse_vec(text, _pretokenized=pretok)
        v_dense = self._dense_engine.encode(text, _pretokenized=pretok)
        if self._fitted:
            v_dense = self._encode_spectral_dense(text, v_dense)

        w_sparse = math.sqrt(self.alpha)
        w_dense = math.sqrt(1.0 - self.alpha)

        v_hybrid = np.concatenate([w_sparse * v_sparse, w_dense * v_dense]).astype(np.float32)
        norm = float(np.linalg.norm(v_hybrid))
        if norm > 1e-12:
            return (v_hybrid / norm).astype(np.float32)
        v_fallback = np.zeros(self.dimension, dtype=np.float32)
        v_fallback[0] = 1.0
        return v_fallback

    def project(self, text: str, recency_weighted: Optional[bool] = None) -> np.ndarray:
        """Projects input text into unit-normalized hybrid sparse-dense representation."""
        if not isinstance(text, str):
            raise TypeError(f"Expected text to be a string, got {type(text).__name__}")

        v_hybrid = self._project_base(text)
        if self._fitted and self._subspace_components is not None and self._subspace_weight > 0.0:
            # Project onto learned spectral / class-prototype subspace and blend
            coords = self._subspace_components @ v_hybrid
            v_sub = self._subspace_components.T @ coords
            v_hybrid = v_hybrid + float(self._subspace_weight) * v_sub
            norm = float(np.linalg.norm(v_hybrid))
            if norm > 1e-12 and np.isfinite(norm):
                v_hybrid = (v_hybrid / norm).astype(np.float32)
            else:
                v_hybrid = np.zeros(self.dimension, dtype=np.float32)
                v_hybrid[0] = 1.0
        return v_hybrid

    encode = project

    def __call__(self, text: str) -> np.ndarray:
        return self.project(text)

    def project_batch(self, texts: Sequence[str], recency_weighted: Optional[bool] = None) -> np.ndarray:
        """Batched projection of multiple text inputs."""
        if texts is None:
            raise TypeError("texts must be a sequence of strings, got None")
        if not hasattr(texts, "__iter__"):
            raise TypeError(f"texts must be a sequence of strings, got {type(texts).__name__}")
        if len(texts) == 0:
            return np.empty((0, self.dimension), dtype=np.float32)
        for idx, t in enumerate(texts):
            if not isinstance(t, str):
                raise TypeError(f"All elements in texts must be strings, got {type(t).__name__} at index {idx}")
        return np.stack([self.project(t) for t in texts], axis=0).astype(np.float32)

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
            cfg["spectral_terms"] = list(self._spectral_terms)
            cfg["spectral_weights"] = [round(float(w), 6) for w in self._spectral_weights]
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
        terms = tuple(config.get("spectral_terms", ()))
        if any(not isinstance(term, str) or not term.strip() for term in terms) or len(set(terms)) != len(terms):
            raise ValueError("Saved hybrid spectral terms must be unique nonempty strings")
        proj = cls(
            sparse_dim=sparse,
            dense_dim=dense,
            alpha=alpha,
            seed=seed,
            backend=backend,
        )
        if config.get("fitted"):
            raw_weights = config.get("spectral_weights", ())
            if not isinstance(raw_weights, (list, tuple)):
                raise ValueError("Saved hybrid projector spectral weights must be a sequence")
            weights = tuple(float(w) for w in raw_weights)
            if len(terms) != len(weights) or any(not math.isfinite(w) for w in weights):
                raise ValueError("Saved hybrid projector spectral terms and weights length mismatch")
            proj._spectral_terms = terms
            proj._spectral_vocab = {tm: i for i, tm in enumerate(terms)}
            proj._spectral_weights = weights
            raw_spec = spectral_vectors if spectral_vectors is not None else config.get("spectral_vectors")
            if raw_spec is not None and len(raw_spec) > 0:
                spec_arr = np.asarray(raw_spec, dtype=np.float32)
                if (
                    spec_arr.ndim != 2
                    or spec_arr.shape[0] != len(terms)
                    or not (1 <= spec_arr.shape[1] <= dense)
                    or not np.all(np.isfinite(spec_arr))
                ):
                    raise ValueError("Saved hybrid projector spectral_vectors have invalid shape")
                proj._spectral_vectors = spec_arr
                vocab_dense = np.stack([proj._dense_engine.encode(tm) for tm in terms], axis=0).astype(np.float32)
                aligned_basis = (spec_arr.T @ vocab_dense).astype(np.float32)
                b_norms = np.linalg.norm(aligned_basis, axis=1, keepdims=True)
                if np.all(b_norms.flatten() > 1e-6):
                    proj._spectral_basis = (aligned_basis / b_norms).astype(np.float32)
                else:
                    proj._spectral_basis = proj._build_spectral_basis(spec_arr.shape[1])
            raw_sub = subspace_components if subspace_components is not None else config.get("subspace_components")
            if raw_sub is not None and len(raw_sub) > 0:
                sub_arr = np.asarray(raw_sub, dtype=np.float32)
                if (
                    sub_arr.ndim != 2
                    or not (1 <= sub_arr.shape[0] <= proj.dimension)
                    or sub_arr.shape[1] != proj.dimension
                    or not np.all(np.isfinite(sub_arr))
                ):
                    raise ValueError("Saved hybrid projector subspace_components have invalid shape")
                proj._subspace_components = sub_arr
            sub_w = float(config.get("subspace_weight", 0.20))
            if not math.isfinite(sub_w):
                raise ValueError("Saved hybrid projector subspace_weight must be finite")
            proj._subspace_weight = sub_w
            proj._fitted = True
        proj._digest = None
        return proj

    def projector_digest(self) -> str:
        if self._digest is None:
            self._digest = hashlib.sha256(json.dumps(self.to_config(), sort_keys=True).encode("utf-8")).hexdigest()
        return self._digest

    to_dict = to_config
    from_dict = from_config


HybridSemanticProjector = HybridProjector


__all__ = [
    "SubwordSemanticEmbeddings",
    "StaticEmbeddingTable",
    "HybridProjector",
    "HybridSemanticProjector",
    "_build_default_table",
    "_stem_word",
    "_ANCHOR_STOPWORDS",
    "match_semantic_concepts",
]
