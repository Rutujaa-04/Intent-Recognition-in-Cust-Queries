"""
Dataset Generator & Loader for Customer Intent Recognition.
Creates an initial realistic multi-class customer query dataset across 10 core business intents.
"""
import json
import random
from typing import Dict, List, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.data.preprocessor import TextPreprocessor


# Seed queries for 10 intents with diverse linguistic patterns
SEED_INTENT_QUERIES: Dict[str, List[str]] = {
    "track_order": [
        "Where is my package right now?",
        "Can you track my shipment for order #98231?",
        "When is my parcel scheduled to arrive?",
        "I have not received my delivery yet, where is it?",
        "Tracking status says in transit but no update for 3 days",
        "Could you check the delivery status for my purchase?",
        "Track order ORD-55412 please",
        "How long until my shoes arrive?",
        "Is there an estimated delivery date for order #77123?",
        "My tracking link shows delivered but I haven't received anything",
        "Where can I find the courier tracking number?",
        "Status of shipment for item ordered yesterday",
        "Delivery delay on my laptop order",
        "Has my order #44901 shipped yet?",
        "Check courier whereabouts for my parcel"
    ],
    "request_refund": [
        "I want a full refund for this damaged shirt",
        "How do I return an item and get my money back?",
        "Can I get a refund? The product arrived broken",
        "I sent back the package, when will my refund be processed?",
        "Requesting reimbursement for order #12890",
        "The item is not what was advertised, please refund me",
        "Where is my refund for the returned headphones?",
        "I received the wrong size and want to return for a refund",
        "Please credit $45 back to my original payment method",
        "What is your return and refund policy?",
        "Initiate return and refund process for my last order",
        "I was promised a refund last week, still waiting",
        "Refund my money immediately for defective goods",
        "Can I return an opened box item for a refund?",
        "How many days does it take to see the refund in my bank?"
    ],
    "cancel_order": [
        "I made a mistake, please cancel order #98124",
        "Can I cancel my order before it ships out?",
        "Please cancel my purchase immediately",
        "I want to stop my order from being delivered",
        "Cancel order ORD-33211 right away",
        "I accidentally placed duplicate orders, please cancel one",
        "Is it too late to cancel my recent purchase?",
        "Stop shipment and cancel my transaction",
        "Need to void my order from 10 minutes ago",
        "Cancel my order and do not charge my card",
        "Please revoke order #88412",
        "I changed my mind, cancel my cart checkout order",
        "How can I cancel an order that is pending confirmation?",
        "Abort delivery and cancel the order",
        "Cancel item #44102 from my basket order"
    ],
    "change_shipping_address": [
        "Can I update my delivery address for order #44123?",
        "I typed the wrong apartment number, please change it",
        "Change my shipping destination to 456 Elm Street",
        "Is it possible to redirect my package to another address?",
        "Update shipping address before courier picks it up",
        "I moved to a new house, need to fix my delivery address",
        "Wrong postal code entered on order ORD-88192",
        "Please change shipping location to my office address",
        "Can I edit the delivery contact phone number and address?",
        "Need to modify address on pending order #11293",
        "Sent to old address by mistake, please correct it",
        "Change receiver address for shipment",
        "I need my package delivered to a different street",
        "How do I amend shipping information after payment?",
        "Update delivery address on file for my current order"
    ],
    "account_issue": [
        "I forgot my account password and cannot log in",
        "How do I reset my password?",
        "Two-factor authentication code is not arriving on my phone",
        "My account has been locked due to multiple attempts",
        "I want to change the email associated with my profile",
        "Cannot sign in to my portal dashboard",
        "How can I update my profile personal details?",
        "My login credentials are not recognized",
        "Please unlock my user account",
        "Verification code expired, send me a new login link",
        "I want to delete my account and user data",
        "Having trouble logging into the mobile app",
        "Reset security questions on my account",
        "Unauthorized login detected on my profile",
        "Can you send a password reset email to my address?"
    ],
    "payment_problem": [
        "My credit card was declined during checkout",
        "I was charged twice for the same transaction",
        "Why is there a duplicate deduction of $89 on my statement?",
        "Payment failed but the amount was deducted from my bank",
        "Which payment methods do you accept?",
        "Can I pay using PayPal or Apple Pay?",
        "Trouble processing payment with my debit card",
        "Billing error on invoice #88921",
        "Card payment error code 403 on payment gateway",
        "Why did my payment get rejected?",
        "I was overcharged on my last purchase",
        "Update my saved credit card for auto-pay",
        "Promo discount coupon code was not applied to payment",
        "Unexpected fee on my checkout total",
        "Bank says payment went through but website shows failed"
    ],
    "product_inquiry": [
        "Does this jacket run true to size?",
        "Is the wireless mouse compatible with Mac and Windows?",
        "When will the black leather boots be back in stock?",
        "What are the exact dimensions of this study desk?",
        "Does this product come with a manufacturer warranty?",
        "Are these wireless headphones water resistant?",
        "Can you tell me the battery life of this model?",
        "Is this item made of genuine leather or synthetic?",
        "Do you have a size chart for children shoes?",
        "Does the camera include memory cards in the box?",
        "Is this phone 5G enabled?",
        "What materials are used in this organic t-shirt?",
        "Can I get recommendations for gaming monitors?",
        "Do you sell replacement parts or cables for this device?",
        "How does product model X compare with model Y?"
    ],
    "subscription_management": [
        "How do I cancel my monthly subscription renewal?",
        "I want to upgrade from basic plan to pro plan",
        "Downgrade my membership to free tier",
        "When does my current annual plan expire?",
        "Stop auto-renewal on my premium membership",
        "Can I pause my subscription for two months?",
        "Billing cycle question regarding my monthly subscription",
        "Change my subscription payment frequency to annual",
        "I was charged for renewal after requesting cancellation",
        "How to switch between subscription tiers?",
        "Cancel trial membership before it charges my account",
        "Add an extra user seat to my business plan",
        "What benefits are included in VIP membership?",
        "Renew my expired subscription with a discount",
        "Remove subscription auto-debit from my account"
    ],
    "contact_human_agent": [
        "Let me speak to a real human person please",
        "Connect me with a customer support representative",
        "I want to talk to an agent, this bot is not helping",
        "Can I speak with a supervisor or manager?",
        "Transfer me to human assistance immediately",
        "Get me a real customer service executive on the line",
        "I need live support right now",
        "Stop giving automated replies, I need a person",
        "Is there a phone number I can call to speak with someone?",
        "Connect to live chat agent",
        "Requesting callback from support specialist",
        "I want to speak with your escalation department",
        "Human representative needed for urgent dispute",
        "Can someone call me back at my phone number?",
        "Transfer this chat to customer service desk"
    ],
    "feedback_or_complaint": [
        "The customer service was extremely rude and unhelpful",
        "My package arrived completely crushed and damaged",
        "Terrible experience with delivery delay of two weeks",
        "I want to file a formal complaint against the courier driver",
        "Great quality product, loved the packaging and fast shipping",
        "Very disappointed with the build quality of this gadget",
        "The app keeps crashing when searching for products, fix it",
        "Worst customer support experience I have ever had",
        "Food order arrived cold and stale, very bad service",
        "I want to submit positive feedback for agent Sarah",
        "Misleading product description on website, very unsatisfied",
        "I have been waiting 40 minutes on hold, awful service",
        "Your website checkout is very frustrating to use",
        "Complain about late delivery and broken promise",
        "Pleased with the quick resolution, thank you team"
    ]
}

# Template expansions to synthesize robust variety (synonyms, conversational prefixes, suffixes)
PREFIXES = [
    "",
    "Hi, ",
    "Hello support, ",
    "Excuse me, ",
    "Please help me, ",
    "Quick question: ",
    "Hey there, ",
    "Can someone tell me, ",
    "Urgent: "
]

SUFFIXES = [
    "",
    " thanks.",
    " please.",
    " as soon as possible.",
    " appreciate any help.",
    " ASAP.",
    " thank you so much."
]


def generate_synthetic_dataset(samples_per_intent: int = 120, random_seed: int = 42) -> pd.DataFrame:
    """
    Augments seed queries with diverse conversational structures to generate
    a comprehensive balanced dataset for training and testing.
    """
    random.seed(random_seed)
    records = []

    for intent, queries in SEED_INTENT_QUERIES.items():
        count = 0
        # Add original queries first
        for q in queries:
            records.append({"query": q, "intent": intent})
            count += 1

        # Synthesize variations with prefixes, suffixes, and minor punctuation changes
        while count < samples_per_intent:
            base_q = random.choice(queries)
            prefix = random.choice(PREFIXES)
            suffix = random.choice(SUFFIXES)

            # Minor variation: lowercased, capitalization, or question marks
            modified_q = f"{prefix}{base_q.strip('.?!')}{suffix}"
            records.append({"query": modified_q, "intent": intent})
            count += 1

    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df


def prepare_and_save_data(random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates dataset, cleans queries, performs stratified 70/15/15 split,
    and writes CSV/JSON files to the project data directory.
    """
    print("[1/4] Generating customer query dataset across 10 business intents...")
    df = generate_synthetic_dataset(samples_per_intent=150, random_seed=random_state)

    # Save raw dataset
    df.to_json(config.RAW_DATASET_FILE, orient="records", indent=2)
    print(f"      Saved raw dataset ({len(df)} samples) to {config.RAW_DATASET_FILE}")

    # Stratified Train/Val/Test Split (70% train, 15% val, 15% test)
    print("[2/4] Splitting into stratified train (70%), val (15%), test (15%)...")
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=random_state, stratify=df["intent"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=random_state, stratify=temp_df["intent"]
    )

    # Clean text columns using TextPreprocessor
    preprocessor = TextPreprocessor()
    for split_df in [train_df, val_df, test_df]:
        split_df["cleaned_query"] = preprocessor.transform(split_df["query"])

    # Save processed splits
    train_df.to_csv(config.TRAIN_DATA_FILE, index=False)
    val_df.to_csv(config.VAL_DATA_FILE, index=False)
    test_df.to_csv(config.TEST_DATA_FILE, index=False)

    print(f"      Train split: {len(train_df)} samples -> {config.TRAIN_DATA_FILE}")
    print(f"      Val split:   {len(val_df)} samples -> {config.VAL_DATA_FILE}")
    print(f"      Test split:  {len(test_df)} samples -> {config.TEST_DATA_FILE}")

    return train_df, val_df, test_df


def load_data(split: str = "train") -> pd.DataFrame:
    """
    Loads saved processed dataset split. Generates if missing.
    """
    file_map = {
        "train": config.TRAIN_DATA_FILE,
        "val": config.VAL_DATA_FILE,
        "test": config.TEST_DATA_FILE
    }
    target_path = file_map.get(split)
    if not target_path or not target_path.exists():
        train_df, val_df, test_df = prepare_and_save_data()
        return {"train": train_df, "val": val_df, "test": test_df}[split]

    return pd.read_csv(target_path)


if __name__ == "__main__":
    prepare_and_save_data()
