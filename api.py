from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import re
import json
import pandas as pd
import unicodedata
import nltk
import logging
from pathlib import Path
from datetime import datetime

from sklearn.base import BaseEstimator, TransformerMixin
from customtransformer import SBERTTransformer, PrototypeSBERTSimTransformer

_HAS_SVD = False
try:
    from sklearn.decomposition import TruncatedSVD
    _HAS_SVD = True
except Exception:
    _HAS_SVD = False

from sentence_transformers import SentenceTransformer
from nltk.stem import WordNetLemmatizer

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths / Config
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
NLTK_DATA_DIR = BASE_DIR / "nltk_data"

BERT_MODEL_PATH = MODEL_DIR / "bert_privacy_model"
ROBERTA_MODEL_PATH = MODEL_DIR / "roberta_privacy_binary"
PRIVACY_LABELS_PATH = MODEL_DIR / "labels.json"
STAGE2_MODEL_PATH = MODEL_DIR / "stage2-model.joblib"

MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 10 * 1024 * 1024))
BERT_THRESHOLD = float(os.getenv("BERT_THRESHOLD", 0.50))

ALLOWED_EXTENSIONS = {"txt", "csv", "xlsx"}
FEEDBACK_CSV_PATH = Path.home() / "privacy_feedback_dataset.csv"

nltk.data.path.append(str(NLTK_DATA_DIR))
lemmatizer = WordNetLemmatizer()

def check_nltk_resource(resource_path, resource_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        raise RuntimeError(
            f"Missing NLTK resource: {resource_name}. "
            f"Install it before running the server."
        )

check_nltk_resource("tokenizers/punkt", "punkt")
check_nltk_resource("corpora/wordnet", "wordnet")

try:
    nltk.data.find("tokenizers/punkt_tab")
except Exception:
    pass

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
CORS(app, resources={r"/*": {"origins": "*"}})

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_text(s):
    s = "" if s is None else str(s)
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fix_unicode(text):
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_tfidf(text):
    text = fix_unicode(text)
    text = re.sub(r"\+?[A-Za-z]\d+:[A-Za-z]\d+", " ", text)
    text = re.sub(r"\b[A-Za-z]\d+\b", " ", text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    tokens = [tok for tok in tokens if len(tok) > 1]
    return " ".join(tokens)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
try:
    privacy_tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_MODEL_PATH))
    privacy_bert_model = AutoModelForSequenceClassification.from_pretrained(str(ROBERTA_MODEL_PATH))
    privacy_bert_model.to(device)
    privacy_bert_model.eval()
except Exception as e:
    raise RuntimeError(f"Failed to load Stage 1 RoBERTa model: {e}")

    
try:
    with open(PRIVACY_LABELS_PATH, "r", encoding="utf-8") as f:
        PRIVACY_LABEL_MAP = json.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load privacy labels: {e}")

logger.info("Stage 1 RoBERTa privacy model loaded successfully")
logger.info("Device: %s", device)
logger.info("Threshold: %s", BERT_THRESHOLD)

STRONG_PRIVACY_PATTERNS = [
    r"\bpersonal data\b",
    r"\bpersonal information\b",
    r"\bsensitive personal information\b",
    r"\bsensitive personal data\b",
    r"\buser data\b",
    r"\baccount data\b",
    r"\bprofile data\b",
    r"\bcustomer data\b",
    r"\bcustomer information\b",
    r"\bcustomer records\b",
    r"\buser records\b",
    r"\bpersonal records\b",
    r"\baccount-related information\b",
    r"\baccount activity\b",
    r"\baccount activity history\b",
    r"\baccount history\b",
    r"\bbrowsing history\b",
    r"\bsearch history\b",
    r"\bclear browsing history\b",
    r"\bclear search history\b",
    r"\bdelete browsing history\b",
    r"\bdelete search history\b",
    r"\bclear account history\b",
    r"\bdelete account history\b",
    r"\bdelete\b",
    r"\bdeletion\b",
    r"\berase\b",
    r"\berasure\b",
    r"\bremove my data\b",
    r"\bremove account data\b",
    r"\bright to delete\b",
    r"\bright to erasure\b",
    r"\bconsent\b",
    r"\bwithdraw consent\b",
    r"\bopt out\b",
    r"\bopt-out\b",
    r"\bdo not sell\b",
    r"\bdo not share\b",
    r"\bright to know\b",
    r"\bright to access\b",
    r"\brequest a copy\b",
    r"\bcopy of my data\b",
    r"\bcopy of personal data\b",
    r"\bexport my data\b",
    r"\bexport personal data\b",
    r"\bdownload my data\b",
    r"\bdownload personal data\b",
    r"\baccess customer records\b",
    r"\baccess personal information\b",
    r"\bview personal data\b",
    r"\bview personal information\b",
    r"\bview customer records\b",
    r"\bview customer information\b",
    r"\bview user data\b",
    r"\bview account-related information\b",
    r"\bobtain personal data\b",
    r"\bsee personal data\b",
    r"\bsee customer information\b",
    r"\bcorrect\b",
    r"\bcorrection\b",
    r"\brectify\b",
    r"\brectification\b",
    r"\bamend personal data\b",
    r"\bupdate inaccurate information\b",
    r"\banonymi[sz]e\b",
    r"\bpseudonymi[sz]e\b",
    r"\bencrypted personal data\b",
    r"\bencrypt personal data\b",
    r"\bdata breach\b",
    r"\bbreach notification\b",
    r"\bretain\b",
    r"\bretention\b",
    r"\bstorage limitation\b",
    r"\bretain personal data\b",
    r"\bretain customer information\b",
    r"\bdata sharing\b",
    r"\bthird parties\b",
    r"\bthird-party\b",
    r"\bgdpr\b",
    r"\bccpa\b",
    r"\baudit logs of access\b",
    r"\baudit logs of access to personal information\b",
    r"\baudit logs of access to customer information\b",
    r"\btemporary account reports\b",
    r"\baccount reports\b",
    r"\bsupport staff should only see\b",
    r"\bonly see the customer information required\b",
    r"\bonly see the information required\b",
    r"\brelevant to their role\b",
    r"\bauthorized staff\b",
    r"\bauthorized personnel\b",
    r"\bauthorized employees\b",
    r"\binformed how .* information .* handled\b",
    r"\binformation is handled\b",
    r"\bhow their information is handled\b",
    r"\bhow customer information is handled\b"
]

SECURITY_ONLY_PATTERNS = [
    r"\bpassword\b",
    r"\bpassword policy\b",
    r"\bpassword policies\b",
    r"\breset password\b",
    r"\bhash(?:ed|ing)?\b",
    r"\b2fa\b",
    r"\btwo-factor\b",
    r"\bmulti-factor\b",
    r"\bauthentication\b",
    r"\blogin\b",
    r"\blog in\b",
    r"\bsign[- ]?in\b",
    r"\bsession\b",
    r"\binactivity\b",
    r"\btimeout\b",
    r"\bexpire after\b",
    r"\bfailed login\b",
    r"\bsuspicious login\b",
    r"\bunauthorized login\b",
    r"\bblock suspicious\b",
    r"\bclient and server\b",
    r"\bencrypt all communication\b",
    r"\bencrypted communication\b",
    r"\bsecure communication\b",
    r"\bbackup\b",
    r"\baudit(?:ing)?\b",
    r"\blog user actions\b",
    r"\bsecurity\b",
    r"\bdetect and block\b",
    r"\bstrong password\b",
    r"\bprivileged-access\b"
]

SYSTEM_ONLY_PATTERNS = [
    r"\bnotifications?\b",
    r"\breceive notifications?\b",
    r"\bgenerate reports?\b",
    r"\breports? based on user activity\b",
    r"\bactivity reports?\b",
    r"\bdark mode\b",
    r"\bshopping cart\b",
    r"\badd items? to cart\b",
    r"\bfilter products?\b",
    r"\bprofile picture\b",
    r"\bnotification settings\b",
    r"\bdisplay notifications\b",
    r"\bshow notifications\b",
    r"\bload within\b",
    r"\bsearch functionality\b",
    r"\bmultiple languages\b",
    r"\btheme\b",
    r"\btheme settings\b",
    r"\binterface theme\b",
    r"\border history\b",
    r"\bloyalty points\b",
    r"\brecommended products\b",
    r"\breorder previous purchases\b"
]

WEAK_PRIVACY_CONTEXT_PATTERNS = [
    r"\buser data\b",
    r"\baccount data\b",
    r"\bactivity data\b",
    r"\bprofile data\b",
    r"\bcustomer information\b",
    r"\bcustomer records\b",
    r"\buser records\b",
    r"\baccount activity\b",
    r"\baccount-related information\b",
    r"\baccount history\b",
    r"\bbrowsing history\b",
    r"\bsearch history\b",
    r"\btemporary account reports\b",
    r"\baccount reports\b",
    r"\baudit logs\b",
    r"\baudit trail\b",
    r"\bsupport staff\b",
    r"\bauthorized staff\b",
    r"\bauthorized personnel\b",
    r"\brelevant to their role\b",
    r"\bonly see .* information required\b",
    r"\bhow .* information .* handled\b",
    r"\binformation is handled\b",
    r"\bhow their information is handled\b",
    r"\bhow customer information is handled\b",
    r"\bview user data\b",
    r"\bview customer information\b",
    r"\bview account-related information\b",
    r"\baccess customer records\b",
    r"\baccess account data\b",
    r"\btrack user activity\b",
    r"\btracking\b",
    r"\bdata use\b",
    r"\bhow .* data .* used\b",
    r"\bclear browsing history\b",
    r"\bclear search history\b",
    r"\bdata collection practices\b",
    r"\bdata sharing preferences\b",
    r"\buser interactions\b",
    r"\buser behavior\b"
]

IMPLICIT_PRIVACY_PATTERNS = [
    r"\bprivacy notice\b",
    r"\bnotice at collection\b",
    r"\bat the time of data collection\b",
    r"\bautomated decision[-\s]?making\b",
    r"\bautomated decisions\b",
    r"\bprofiling\b",
    r"\bprofile users\b",
    r"\bpersonalize(?:d|ation)?\b",
    r"\bpersonalized ads?\b",
    r"\buser behavior\b",
    r"\bbehavioral tracking\b",
    r"\bbrowsing patterns?\b",
    r"\binferred user interests?\b",
    r"\binfer user preferences\b",
    r"\bmetadata\b",
    r"\buser interactions\b",
    r"\blocation data\b",
    r"\btrack user location\b",
    r"\bvisible to others\b",
    r"\bdata visibility\b",
    r"\buser roles\b",
    r"\brole[-\s]?based visibility\b",
    r"\bretention period\b",
    r"\bstorage period\b",
    r"\bkept indefinitely\b",
    r"\blogs? of user interactions\b",
    r"\bdata collection practices\b",
    r"\bdata sharing preferences\b",
    r"\bcontrol their data sharing\b",
    r"\bprivacy by design\b",
    r"\bprivacy by default\b",
    r"\bby design and by default\b"
]

FORCE_PRIVACY_PATTERNS = [
    r"\bprivacy notice\b",
    r"\bnotice at collection\b",
    r"\bat the time of data collection\b",
    r"\bautomated decision[-\s]?making\b",
    r"\bopt out of automated decision[-\s]?making\b",
    r"\bdata protection by design\b",
    r"\bdata protection by default\b",
    r"\bprivacy by design\b",
    r"\bprivacy by default\b",
    r"\bby design and by default\b",
    r"\bretention period\b",
    r"\bremove outdated personal data\b",
    r"\boutdated personal data\b",
    r"\bkept indefinitely\b",
    r"\blogs? of user interactions indefinitely\b",
    r"\bvisible to others\b",
    r"\bdata visibility\b",
    r"\blimit data visibility\b",
    r"\brole[-\s]?based visibility\b",
    r"\binfer user preferences\b",
    r"\bbrowsing patterns?\b",
    r"\binferred user interests?\b",
    r"\bpersonalize(?:d|ation)?\b",
    r"\buser behavior\b",
    r"\blocation data\b"
]

def matches_any_pattern(text, patterns):
    text_l = str(text).lower()
    return any(re.search(p, text_l) for p in patterns)

def privacy_signal_type(text):
    text_l = str(text).lower()

    has_strong = matches_any_pattern(text_l, STRONG_PRIVACY_PATTERNS)
    has_weak = matches_any_pattern(text_l, WEAK_PRIVACY_CONTEXT_PATTERNS)
    has_implicit = matches_any_pattern(text_l, IMPLICIT_PRIVACY_PATTERNS)
    has_security_only = matches_any_pattern(text_l, SECURITY_ONLY_PATTERNS)
    has_system_only = matches_any_pattern(text_l, SYSTEM_ONLY_PATTERNS)

    if has_strong:
        return "strong_privacy"

    if has_implicit:
        return "weak_privacy"

    if has_weak and not (has_security_only or has_system_only):
        return "weak_privacy"

    return "no_privacy_signal"

def predict_privacy_prob_bert(text):
    text = clean_text(text)

    inputs = privacy_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = privacy_bert_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)

    return float(probs[0][1])

def predict_privacy_gate(text):
    text = clean_text(text)

    prob = predict_privacy_prob_bert(text)

    signal_type = privacy_signal_type(text)
    force_privacy = matches_any_pattern(text, FORCE_PRIVACY_PATTERNS)
    has_security_only = matches_any_pattern(text, SECURITY_ONLY_PATTERNS)
    has_system_only = matches_any_pattern(text, SYSTEM_ONLY_PATTERNS)
    has_strong_privacy = matches_any_pattern(text, STRONG_PRIVACY_PATTERNS)

    extra_security_only = bool(re.search(
        r"\b("
        r"two-factor authentication|2fa|multi-factor authentication|"
        r"strong password policies|password policy|password policies|"
        r"login attempts|failed login attempts|suspicious login attempts|"
        r"session[s]? shall expire|expire after a period of inactivity|"
        r"client and server|encrypt all communication|encrypted communication|"
        r"block suspicious login attempts|authentication for user login|"
        r"user login|login security|password reset|hash passwords|"
        r"secure communication|session timeout|inactivity timeout"
        r")\b",
        text,
        re.I
    ))

    extra_system_only = bool(re.search(
        r"\b("
        r"display notifications to users|show notifications|notifications to users|"
        r"multiple languages|shopping cart|theme of the interface|"
        r"change the theme|load within two seconds"
        r")\b",
        text,
        re.I
    ))

    non_privacy_action = bool(re.search(
        r"\b("
        r"download invoices|download invoice|invoice|receipt|"
        r"shopping cart|cart items|items from their shopping cart|"
        r"export sales reports|sales reports|analytics report|"
        r"product|products|catalog|order|orders"
        r")\b",
        text,
        re.I
    ))

    has_personal_context = matches_any_pattern(text, [
        r"\bpersonal data\b",
        r"\bpersonal information\b",
        r"\buser data\b",
        r"\baccount data\b",
        r"\bdata about me\b",
        r"\bmy data\b"
    ])

    if force_privacy:
        final_label = 1
        decision_source = "rule-based-privacy-override"

    elif (has_security_only or extra_security_only) and not has_strong_privacy:
        final_label = 0
        decision_source = "security-override"

    elif (has_system_only or extra_system_only) and signal_type == "no_privacy_signal":
        final_label = 0
        decision_source = "system-override"

    elif non_privacy_action and not has_personal_context:
        final_label = 0
        decision_source = "non-privacy-action-override"

    else:
        final_label = 1 if prob >= BERT_THRESHOLD else 0
        decision_source = "BERT-threshold"

        if final_label == 0 and signal_type == "weak_privacy" and prob >= 0.35:
            final_label = 1
            decision_source = "BERT+weak-signal"

    logger.info("TEXT: %s", text)
    logger.info("PROB: %s", prob)
    logger.info("SIGNAL: %s", signal_type)
    logger.info("FORCE: %s", force_privacy)
    logger.info("HAS_STRONG_PRIVACY: %s", has_strong_privacy)
    logger.info("SECURITY_ONLY: %s", has_security_only or extra_security_only)
    logger.info("SYSTEM_ONLY: %s", has_system_only or extra_system_only)
    logger.info("FINAL: %s", final_label)
    logger.info("SOURCE: %s", decision_source)
    logger.info("%s", "-" * 80)

    return {
        "final_label": int(final_label),
        "final_name": PRIVACY_LABEL_MAP[str(final_label)],
        "decision_source": decision_source,
        "privacy_prob": float(prob),
        "privacy_signal_type": signal_type,
        "force_privacy": bool(force_privacy)
    }

import __main__

__main__.SBERTTransformer = SBERTTransformer
__main__.PrototypeSBERTSimTransformer = PrototypeSBERTSimTransformer

try:
    bundle = joblib.load(STAGE2_MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load Stage 2 model bundle: {e}")

stage2_model = bundle["model"]
stage2_labels = bundle["labels"]
label_to_i = bundle.get("label_to_i", {lab: i for i, lab in enumerate(stage2_labels)})
cfg = bundle.get("config", {})

logger.info("Stage 2 matcher loaded successfully")
logger.info("Stage 2 labels: %s", len(stage2_labels))

PRED_MODE = cfg.get("PRED_MODE", "TOPK")
TOPK = cfg.get("TOPK", 1)
THRESH = cfg.get("THRESH", 0.0)

GAP_TH = cfg.get("GAP_TH", 0.18)
MIN_SCORE_2 = cfg.get("MIN_SCORE_2", 0.00)
MARGIN_TH = cfg.get("MARGIN_TH", 0.15)
LOWCONF_TOP1 = cfg.get("LOWCONF_TOP1", 0.18)
MIN_TOP1 = cfg.get("MIN_TOP1", -0.25)

MIN_SCORE_3 = cfg.get("MIN_SCORE_3", -0.40)
MARGIN_TH_3 = cfg.get("MARGIN_TH_3", 0.20)

ENABLE_DELETE_LABEL_COUPLING = cfg.get("ENABLE_DELETE_LABEL_COUPLING", True)

DESIGN_RE = re.compile(
    r"\b("
    r"privacy by design|privacy by default|data protection by design|data protection by default|"
    r"by design and by default|not publicly visible by default|private by default|"
    r"privacy-friendly defaults|default settings should favor confidentiality|"
    r"sharing disabled unless|disabled unless the user actively enables"
    r")\b",
    re.I
)

DEL_RE = re.compile(
    r"\b(delete|deletion|erase|erasure|deleted|remove my data|remove account data)\b",
    re.I
)

RECTIFY_RE = re.compile(
    r"\b("
    r"correct|correction|rectify|rectification|update inaccurate|"
    r"inaccurate personal information|incomplete personal data"
    r")\b",
    re.I
)

OPTOUT_RE = re.compile(
    r"\b("
    r"opt[\s\-]?out|do not sell|do not share|sale of personal information|"
    r"sharing of personal information|targeted advertising|direct marketing"
    r")\b",
    re.I
)

PORTABILITY_RE = re.compile(
    r"\b("
    r"machine-readable|structured, commonly used|structured, machine-readable|"
    r"portable export|data portability|transmit.*another controller|"
    r"another controller|another provider|another service"
    r")\b",
    re.I
)

ARTICLE13_RE = re.compile(
    r"\b("
    r"inform|informed|notice at collection|privacy notice|at collection|"
    r"before collection|before collecting|purposes of data collection|"
    r"purpose of data collection|categories of personal information|"
    r"purposes for collecting|disclosed to external processors"
    r")\b",
    re.I
)

ARTICLE22_RE = re.compile(
    r"\b("
    r"automated decision[-\s]?making|automated decisions|profiling|"
    r"decisions made solely by automated processing|explain automated decisions|"
    r"denied access .* automatically"
    r")\b",
    re.I
)

ARTICLE32_RE = re.compile(
    r"\b("
    r"authorized personnel|authorized staff|authorized employees|"
    r"restrict access|access control|encrypt|encrypted|encryption|"
    r"audit trail|audit log|auditing|security team|unauthorized modification|"
    r"personal data breach|data breach|breach notification|"
    r"suspicious data access patterns|segment internal networks|"
    r"customer records|personal records"
    r")\b",
    re.I
)

INTERNAL_ACCESS_RE = re.compile(
    r"\b("
    r"administrator|administrators|admin|admins|employee|employees|staff|"
    r"support staff|authorized personnel|authorized staff|authorized employees|"
    r"troubleshooting|auditing|audit trail|security team|internal"
    r")\b",
    re.I
)

USER_ACCESS_RIGHT_RE = re.compile(
    r"\b("
    r"users shall be able to access|users shall be able to download|"
    r"download a copy|copy of all personal data|copy of personal data|"
    r"obtain confirmation|receive their personal data|"
    r"stored about them|personal data associated with their account|"
    r"access all personal data stored about them|copy of their stored personal data"
    r")\b",
    re.I
)

ARTICLE15_RE = re.compile(
    r"\b("
    r"copy of personal data|download a copy|obtain confirmation|"
    r"access all personal data stored about them|copy of their stored personal data|"
    r"view personal data|receive their personal data|data stored about them"
    r")\b",
    re.I
)

ARTICLE5_RE = re.compile(
    r"\b("
    r"only necessary personal data|only what is necessary|limited to what is necessary|"
    r"retention period|no longer than necessary|as long as necessary|"
    r"storage limitation|keep .* only for the required retention period"
    r")\b",
    re.I
)

# Helpers
def get_scores(model_obj, texts):
    sc = np.asarray(model_obj.decision_function(list(texts)))
    if sc.ndim == 1:
        sc = sc.reshape(-1, 1)
    return sc

def label_score(scores_row, lab):
    i = label_to_i.get(lab)
    if i is None:
        return -999.0
    return float(scores_row[i])

def set_label(row, lab):
    i = label_to_i.get(lab)
    if i is not None:
        row[i] = 1

def unset_label(row, lab):
    i = label_to_i.get(lab)
    if i is not None:
        row[i] = 0

def has_label(row, lab):
    i = label_to_i.get(lab)
    if i is None:
        return False
    return bool(row[i] == 1)

def dedupe_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out

def _set_many(row, *labels):
    for lab in labels:
        set_label(row, lab)

def _unset_many(row, *labels):
    for lab in labels:
        unset_label(row, lab)

def _text_has(text, pattern):
    return bool(pattern.search(text))

# Split compound requirements
def split_compound_requirement(text):
    text = clean_text(text)
    if not text:
        return []

    original_text = text.rstrip(".")
    prefix = ""
    remainder = original_text

    prefix_patterns = [
        r"^(As\s+an?\s+[^,]+,\s*I\s+want\s+to\s+)(.+)$",
        r"^(As\s+an?\s+[^,]+,\s*I\s+need\s+to\s+)(.+)$",
        r"^(I\s+want\s+to\s+)(.+)$",
        r"^(I\s+need\s+to\s+)(.+)$",
        r"^(The\s+system\s+shall\s+)(.+)$",
        r"^(The\s+user\s+can\s+)(.+)$",
        r"^(Users\s+can\s+)(.+)$",
    ]

    for pat in prefix_patterns:
        m = re.match(pat, original_text, flags=re.I)
        if m:
            prefix = m.group(1).strip()
            remainder = m.group(2).strip()
            break

    temp = remainder
    temp = re.sub(r"\s*,\s*and\s+", ", ", temp, flags=re.I)
    temp = re.sub(r"\s*,\s*or\s+", ", ", temp, flags=re.I)
    temp = re.sub(r"\s+and\s+", ", ", temp, flags=re.I)
    temp = re.sub(r"\s+or\s+", ", ", temp, flags=re.I)
    temp = re.sub(r"\s*;\s*", ", ", temp)
    temp = re.sub(r"\s+", " ", temp).strip(" ,")

    parts = [p.strip(" ,.") for p in temp.split(",") if p.strip(" ,.")]
    if len(parts) <= 1:
        return [text]

    shared_tail = None

    tokenized_parts = [p.split() for p in parts]
    if len(tokenized_parts) >= 2:
        shortest_len = min(len(tp) for tp in tokenized_parts)
        common_suffix = []

        for i in range(1, shortest_len + 1):
            candidate = tokenized_parts[0][-i:]
            if all(tp[-i:] == candidate for tp in tokenized_parts if len(tp) >= i):
                common_suffix = candidate
            else:
                break

        if common_suffix and len(common_suffix) >= 1:
            shared_tail = " ".join(common_suffix).strip()

    if not shared_tail:
        longest = max(parts, key=lambda x: len(x.split()))
        longest_tokens = longest.split()
        if len(longest_tokens) >= 2:
            shared_tail = " ".join(longest_tokens[1:]).strip()

    sub_requirements = []

    for part in parts:
        clause_body = part.strip()

        if shared_tail:
            part_tokens = clause_body.split()
            if len(part_tokens) == 1:
                clause_body = f"{clause_body} {shared_tail}"

        if shared_tail:
            if clause_body.lower() == shared_tail.lower():
                continue

        if prefix:
            clause = f"{prefix} {clause_body}".strip()
        else:
            clause = clause_body

        clause = re.sub(r"\s+", " ", clause).strip()
        if not clause.endswith("."):
            clause += "."

        sub_requirements.append(clause)

    sub_requirements = dedupe_preserve_order(sub_requirements)
    cleaned_subs = [s.strip() for s in sub_requirements if s.strip()]

    if len(cleaned_subs) <= 1:
        return [text]

    return cleaned_subs

def collect_label_evidence(text, matches=None):
    text = clean_text(text)
    evidence = {}

    def add_evidence(label, patterns):
        if matches is not None and label not in matches:
            return

        if not isinstance(patterns, list):
            patterns = [patterns]

        found = []
        for pat in patterns:
            for m in pat.finditer(text):
                phrase = clean_text(m.group(0))
                if phrase:
                    found.append(phrase)

        found = dedupe_preserve_order(found)
        if found:
            evidence[label] = found

    add_evidence("gdpr-article 17", DEL_RE)
    add_evidence("ccpa-right to delete", DEL_RE)

    add_evidence("gdpr-article 16", RECTIFY_RE)
    add_evidence("ccpa-right to correct", RECTIFY_RE)

    add_evidence("gdpr-article 15", ARTICLE15_RE)
    add_evidence("ccpa-right to access", ARTICLE15_RE)

    add_evidence("gdpr-article 20", PORTABILITY_RE)

    add_evidence("gdpr-article 21", OPTOUT_RE)
    add_evidence("ccpa-right to opt-out of sale or sharing", OPTOUT_RE)

    add_evidence("gdpr-article 13", ARTICLE13_RE)
    add_evidence("ccpa-notice at collection & non-discrimination", ARTICLE13_RE)

    add_evidence("gdpr-article 22", ARTICLE22_RE)
    add_evidence("gdpr-article 25", DESIGN_RE)
    add_evidence("gdpr-article 32", ARTICLE32_RE)
    add_evidence("gdpr-article 5", ARTICLE5_RE)

    return evidence

def pred_smart_dyn_topk(scores_row, text, allow_top3=False):
    y = np.zeros((len(stage2_labels),), dtype=np.int32)
    order = np.argsort(scores_row)[::-1]

    j1 = int(order[0])
    s1 = float(scores_row[j1])

    if s1 < float(MIN_TOP1):
        return y

    y[j1] = 1

    j2 = None
    for jj in order:
        jj = int(jj)
        if jj != j1:
            j2 = jj
            break

    if j2 is None:
        return y

    s2 = float(scores_row[j2])
    gap12 = s1 - s2

    ambiguous2 = (
        (gap12 <= float(MARGIN_TH)) or
        (s1 <= float(LOWCONF_TOP1) and gap12 <= float(GAP_TH))
    )
    allow2 = ambiguous2 and (s2 >= float(MIN_SCORE_2))

    if allow2:
        y[j2] = 1

    if allow_top3:
        j3 = None
        for jj in order:
            jj = int(jj)
            if jj != j1 and jj != j2:
                j3 = jj
                break

        if j3 is not None:
            s3 = float(scores_row[j3])
            gap23 = s2 - s3
            allow3 = (y.sum() >= 2) and (gap23 <= float(MARGIN_TH_3)) and (s3 >= float(MIN_SCORE_3))
            if allow3:
                y[j3] = 1

    return y

def apply_rule_refinement(text, y_row, scores_row):
    row = y_row.copy()
    t = text or ""

    has_delete = _text_has(t, DEL_RE)
    has_rectify = _text_has(t, RECTIFY_RE)
    has_optout = _text_has(t, OPTOUT_RE)
    has_portability = _text_has(t, PORTABILITY_RE)
    has_notice = _text_has(t, ARTICLE13_RE)
    has_auto = _text_has(t, ARTICLE22_RE)
    has_design = _text_has(t, DESIGN_RE)
    has_security = _text_has(t, ARTICLE32_RE)
    has_internal_access = _text_has(t, INTERNAL_ACCESS_RE)
    has_user_access_right = _text_has(t, USER_ACCESS_RIGHT_RE)
    has_article15 = _text_has(t, ARTICLE15_RE)
    has_article5 = _text_has(t, ARTICLE5_RE)

    if has_delete:
        _unset_many(row, "gdpr-article 15", "ccpa-right to access")
        _set_many(row, "gdpr-article 17", "ccpa-right to delete")

    if has_rectify:
        _unset_many(row, "gdpr-article 15", "gdpr-article 17", "ccpa-right to delete")
        _set_many(row, "gdpr-article 16", "ccpa-right to correct")

    if has_optout:
        _unset_many(row, "gdpr-article 15", "ccpa-right to access")
        _set_many(row, "gdpr-article 21", "ccpa-right to opt-out of sale or sharing")

    if has_portability:
        _unset_many(row, "gdpr-article 15", "ccpa-right to access")
        _set_many(row, "gdpr-article 20")

    if has_notice:
        _set_many(row, "gdpr-article 13", "ccpa-notice at collection & non-discrimination")

    if has_auto:
        _set_many(row, "gdpr-article 22")

    if has_design:
        _set_many(row, "gdpr-article 25")

    if has_article5:
        _set_many(row, "gdpr-article 5")

    if has_security:
        _set_many(row, "gdpr-article 32")

    if has_internal_access and not has_user_access_right:
        _unset_many(row, "gdpr-article 15", "ccpa-right to access")
        if has_security:
            _set_many(row, "gdpr-article 32")

    if has_article15 and has_user_access_right:
        _set_many(row, "gdpr-article 15", "ccpa-right to access")

    return row

def apply_label_coupling(y_row):
    row = y_row.copy()

    if not ENABLE_DELETE_LABEL_COUPLING:
        return row

    coupling_groups = [
        ["gdpr-article 15", "ccpa-right to access"],
        ["gdpr-article 16", "ccpa-right to correct"],
        ["gdpr-article 17", "ccpa-right to delete"],
        ["gdpr-article 21", "ccpa-right to opt-out of sale or sharing"]
    ]

    for group in coupling_groups:
        indices = [label_to_i.get(label) for label in group if label_to_i.get(label) is not None]
        if not indices:
            continue

        if any(row[i] == 1 for i in indices):
            for i in indices:
                row[i] = 1

    return row

def predict_stage2_labels(scores_mat, texts):
    Y_pred = np.zeros((len(texts), len(stage2_labels)), dtype=np.int32)
    allow_top3 = (PRED_MODE.upper() == "SMART_DYN_TOPK_3")

    for i in range(len(texts)):
        srow = scores_mat[i]
        t = texts[i] or ""

        if PRED_MODE.upper().startswith("SMART_DYN_TOPK"):
            y0 = pred_smart_dyn_topk(srow, t, allow_top3=allow_top3)
        elif PRED_MODE.upper() == "TOPK":
            y0 = np.zeros((len(stage2_labels),), dtype=np.int32)
            idx = np.argsort(srow)[::-1][:TOPK]
            y0[idx] = 1
        else:
            y0 = (srow >= THRESH).astype(np.int32)

        y1 = apply_rule_refinement(t, y0, srow)
        y2 = apply_label_coupling(y1)

        if y2.sum() == 0:
            j1 = int(np.argsort(srow)[::-1][0])
            y2[j1] = 1
            y2 = apply_label_coupling(y2)

        Y_pred[i] = y2

    return Y_pred

#Processing helpers
def normalize_override_label(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return 1 if value else 0

    if isinstance(value, int):
        return 1 if value == 1 else 0

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "privacy", "private", "yes", "true"}:
            return 1
        if v in {"0", "non-privacy", "nonprivacy", "no", "false"}:
            return 0

    return None

def build_stage1_result(text, override_label=None):
    text = clean_text(text)
    privacy_result = predict_privacy_gate(text)

    model_label = int(privacy_result["final_label"])
    final_override = normalize_override_label(override_label)

    if final_override is None:
        final_label = model_label
        was_modified = False
    else:
        final_label = final_override
        was_modified = (final_label != model_label)

    final_name = PRIVACY_LABEL_MAP.get(str(final_label), "PRIVACY" if final_label == 1 else "NON-PRIVACY")

    if final_label == 1:
        user_prompt = "This requirement appears to be privacy-related. Do you want to proceed to legal analysis or make modifications?"
    else:
        user_prompt = "This requirement appears to be non-privacy. Do you want to keep this classification or modify it?"

    return {
        "model_prediction": {
            "label": model_label,
            "label_name": privacy_result["final_name"],
            "confidence": float(privacy_result["privacy_prob"]),
            "decision_source": privacy_result["decision_source"],
            "privacy_signal_type": privacy_result.get("privacy_signal_type"),
            "force_privacy": bool(privacy_result.get("force_privacy", False))
        },
        "final_decision": {
            "label": final_label,
            "label_name": final_name,
            "was_modified_by_user": was_modified
        },
        "can_proceed_to_stage2": bool(final_label == 1),
        "user_prompt": user_prompt
    }

def run_stage2_only_single(text):
    text = clean_text(text)

    scores = get_scores(stage2_model, [text])
    Y_pred = predict_stage2_labels(scores, [text])

    idx = list(np.where(Y_pred[0] == 1)[0])
    matches = [stage2_labels[j] for j in idx]

    order = np.argsort(scores[0])[::-1][:5]
    top_scores = [
        {
            "label": stage2_labels[j],
            "score": float(scores[0][j])
        }
        for j in order
    ]

    evidence = collect_label_evidence(text, matches=matches)

    return {
        "status": "completed",
        "matches": matches,
        "match_count": len(matches),
        "top_scores": top_scores,
        "evidence": evidence
    }

def run_stage2_analysis(text, decompose=True):
    text = clean_text(text)
    sub_requirements = split_compound_requirement(text) if decompose else [text]

    if len(sub_requirements) <= 1:
        result = run_stage2_only_single(text)
        result["decomposed"] = False
        result["sub_requirements"] = []
        return result

    sub_results = []
    all_matches = []
    seen = set()
    aggregated_evidence = {}

    for sr in sub_requirements:
        sr_stage2 = run_stage2_only_single(sr)

        sub_results.append({
            "input_text": sr,
            "matches": sr_stage2["matches"],
            "match_count": sr_stage2["match_count"],
            "top_scores": sr_stage2["top_scores"],
            "evidence": sr_stage2["evidence"]
        })

        for m in sr_stage2["matches"]:
            if m not in seen:
                seen.add(m)
                all_matches.append(m)

        for label, phrases in sr_stage2.get("evidence", {}).items():
            aggregated_evidence.setdefault(label, [])
            aggregated_evidence[label].extend(phrases)

    aggregated_evidence = {
        label: dedupe_preserve_order(phrases)
        for label, phrases in aggregated_evidence.items()
    }

    return {
        "status": "completed",
        "decomposed": True,
        "sub_requirements": sub_results,
        "matches": all_matches,
        "match_count": len(all_matches),
        "top_scores": [],
        "evidence": aggregated_evidence
    }

def process_requirement_for_ui(text, override_label=None, proceed_to_stage2=False, decompose=True):
    text = clean_text(text)
    stage1 = build_stage1_result(text, override_label=override_label)

    if stage1["final_decision"]["label"] == 0:
        stage2 = {
            "status": "skipped",
            "reason": "This requirement is currently marked as non-privacy, so Stage 2 was not run.",
            "matches": [],
            "match_count": 0,
            "top_scores": [],
            "evidence": {},
            "decomposed": False,
            "sub_requirements": []
        }
    elif not proceed_to_stage2:
        stage2 = {
            "status": "awaiting_confirmation",
            "reason": "Stage 1 marked this requirement as privacy-related. Waiting for user confirmation before running Stage 2.",
            "matches": [],
            "match_count": 0,
            "top_scores": [],
            "evidence": {},
            "decomposed": False,
            "sub_requirements": []
        }
    else:
        stage2 = run_stage2_analysis(text, decompose=decompose)

    return {
        "input_text": text,
        "stage1": stage1,
        "stage2": stage2
    }

def get_stage1_results(texts, decompose=False):
    texts = [clean_text(t) for t in texts if clean_text(t)]
    results = []

    for text in texts:
        if decompose:
            sub_requirements = split_compound_requirement(text)
            if len(sub_requirements) > 1:
                sub_stage1 = [build_stage1_result(sr) for sr in sub_requirements]
                overall_privacy = any(sr["final_decision"]["label"] == 1 for sr in sub_stage1)

                results.append({
                    "input_text": text,
                    "decomposed": True,
                    "sub_requirements": [
                        {
                            "input_text": sr_text,
                            "stage1": sr_result
                        }
                        for sr_text, sr_result in zip(sub_requirements, sub_stage1)
                    ],
                    "stage1": {
                        "model_prediction": {
                            "label": 1 if overall_privacy else 0,
                            "label_name": "PRIVACY" if overall_privacy else "NON-PRIVACY",
                            "confidence": None,
                            "decision_source": "sub-requirement-aggregation",
                            "privacy_signal_type": None,
                            "force_privacy": False
                        },
                        "final_decision": {
                            "label": 1 if overall_privacy else 0,
                            "label_name": "PRIVACY" if overall_privacy else "NON-PRIVACY",
                            "was_modified_by_user": False
                        },
                        "can_proceed_to_stage2": bool(overall_privacy),
                        "user_prompt": (
                            "This requirement contains multiple clauses. Review the privacy classification and choose whether to proceed."
                        )
                    },
                    "stage2": {
                        "status": "awaiting_confirmation" if overall_privacy else "skipped",
                        "reason": (
                            "Waiting for user confirmation before running Stage 2."
                            if overall_privacy else
                            "This requirement is currently marked as non-privacy, so Stage 2 was not run."
                        ),
                        "matches": [],
                        "match_count": 0,
                        "top_scores": [],
                        "evidence": {},
                        "decomposed": False,
                        "sub_requirements": []
                    }
                })
                continue

        results.append(process_requirement_for_ui(text, override_label=None, proceed_to_stage2=False, decompose=False))

    return results

def extract_texts_from_df(df):
    possible_cols = ["requirement_text", "text", "requirement", "user_story", "sentence"]
    text_col = next((c for c in possible_cols if c in df.columns), None)

    if text_col is None and len(df.columns) > 0:
        text_col = df.columns[0]

    if text_col is None:
        return []

    texts = df[text_col].fillna("").astype(str).str.strip().tolist()
    return [t for t in texts if t]

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"error": "Uploaded file is too large"}), 413

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.exception("Unhandled server error: %s", error)
    return jsonify({"error": "Internal server error"}), 500

# Routes
@app.route("/", methods=["GET"])
def home():
    return "Privacy Requirement Matcher API is running."

@app.route("/save-feedback", methods=["POST"])
def save_feedback():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    text = clean_text(data.get("text", ""))
    user_label = normalize_override_label(data.get("user_label"))
    model_label = normalize_override_label(data.get("model_label"))

    if not text:
        return jsonify({"error": "Missing requirement text"}), 400

    if user_label is None:
        return jsonify({"error": "Invalid user_label. Use 0 for non-privacy or 1 for privacy."}), 400

    FEEDBACK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "text": text,
        "model_label": model_label if model_label is not None else "",
        "user_label": user_label,
        "was_corrected": int(model_label is not None and model_label != user_label)
    }

    df = pd.DataFrame([row])

    file_exists = FEEDBACK_CSV_PATH.exists()

    df.to_csv(
        FEEDBACK_CSV_PATH,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )

    return jsonify({
        "status": "saved",
        "message": "Feedback saved successfully",
        "saved_to": str(FEEDBACK_CSV_PATH.name)
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(device),
        "stage1_loaded": True,
        "stage2_loaded": True
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    texts = []
    override_label = data.get("override_label", None)
    proceed_to_stage2 = bool(data.get("proceed_to_stage2", False))
    decompose = bool(data.get("decompose", True))

    if "text" in data:
        text = str(data["text"]).strip()
        if text:
            texts = [text]

    elif "texts" in data:
        if not isinstance(data["texts"], list):
            return jsonify({"error": "'texts' must be a list"}), 400
        texts = [str(t).strip() for t in data["texts"] if str(t).strip()]

    if not texts:
        return jsonify({"error": "No valid input text provided"}), 400

    if len(texts) == 1:
        result = process_requirement_for_ui(
            texts[0],
            override_label=override_label,
            proceed_to_stage2=proceed_to_stage2,
            decompose=decompose
        )
        return jsonify(result)

    results = get_stage1_results(texts, decompose=False)
    return jsonify({"results": results})

@app.route("/analyze-stage2", methods=["POST"])
def analyze_stage2():
    data = request.get_json(silent=True)

    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    text = str(data.get("text", "")).strip()
    if not text:
        return jsonify({"error": "No valid input text provided"}), 400

    override_label = data.get("override_label", None)
    decompose = bool(data.get("decompose", True))
    force_stage2 = bool(data.get("force_stage2", False))

    stage1 = build_stage1_result(text, override_label=override_label)

    if stage1["final_decision"]["label"] == 0 and not force_stage2:
        return jsonify({
            "input_text": text,
            "stage1": stage1,
            "stage2": {
                "status": "skipped",
                "reason": "This requirement is currently marked as non-privacy, so Stage 2 was not run.",
                "matches": [],
                "match_count": 0,
                "top_scores": [],
                "evidence": {},
                "decomposed": False,
                "sub_requirements": []
            }
        })

    stage2 = run_stage2_analysis(text, decompose=decompose)

    return jsonify({
        "input_text": text,
        "stage1": stage1,
        "stage2": stage2
    })

@app.route("/predict-file", methods=["POST"])
def predict_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if not file or not file.filename:
        return jsonify({"error": "Invalid file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Use .txt, .csv, or .xlsx"}), 400

    filename = file.filename.lower()

    try:
        if filename.endswith(".txt"):
            content = file.read().decode("utf-8", errors="ignore")
            texts = [line.strip() for line in content.splitlines() if line.strip()]

        elif filename.endswith(".csv"):
            df = pd.read_csv(file)
            texts = extract_texts_from_df(df)

        elif filename.endswith(".xlsx"):
            df = pd.read_excel(file)
            texts = extract_texts_from_df(df)

        else:
            return jsonify({"error": "Unsupported file type. Use .txt, .csv, or .xlsx"}), 400

        if not texts:
            return jsonify({"error": "No valid requirements found in file"}), 400

        results = get_stage1_results(texts, decompose=False)

        return jsonify({
            "results": results,
            "message": "Stage 1 completed. Review the classification of each requirement before proceeding to Stage 2."
        })

    except Exception as e:
        logger.exception("File processing failed: %s", e)
        return jsonify({"error": f"File processing failed: {str(e)}"}), 500

if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    app.run(host=host, port=port, debug=False)
