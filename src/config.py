"""
Configuration settings for the Intent Recognition project.
"""
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure runtime directories exist
for path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset file paths
RAW_DATASET_FILE = RAW_DATA_DIR / "customer_queries.json"
TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_FILE = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_FILE = PROCESSED_DATA_DIR / "test.csv"

# Saved Model paths
BASELINE_MODEL_PATH = MODELS_DIR / "tfidf_intent_model.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# Supported Customer Service Intent Taxonomy
INTENT_DEFINITIONS = {
    "track_order": "Checking status, tracking number, package delivery whereabouts.",
    "request_refund": "Return requests, refund status, money back inquiries.",
    "cancel_order": "Cancelling placed or pending orders.",
    "change_shipping_address": "Updating delivery address or contact details before shipment.",
    "account_issue": "Password reset, 2FA, login problems, account settings.",
    "payment_problem": "Failed payments, double charging, card declined, billing questions.",
    "product_inquiry": "Product specs, availability, sizing, stock questions.",
    "subscription_management": "Plan renewal, cancellation, upgrading or downgrading plans.",
    "contact_human_agent": "Explicit requests to talk to a live agent or supervisor.",
    "feedback_or_complaint": "Complaints regarding poor service, damaged items, delays, or feedback."
}

INTENT_LABELS = list(INTENT_DEFINITIONS.keys())

# Fallback & Routing Thresholds
# If confidence is lower than this threshold, route query to human support
CONFIDENCE_THRESHOLD = 0.55
FALLBACK_INTENT = "fallback_unknown"
