"""
Text Preprocessing Pipeline for Customer Queries.
"""
import re
from typing import List, Union
import pandas as pd


# Common English contractions dictionary
CONTRACTION_MAP = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have"
}

_CONTRACTION_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(key) for key in CONTRACTION_MAP.keys()) + r')\b',
    flags=re.IGNORECASE
)


class TextPreprocessor:
    """
    Cleans and normalizes customer queries for NLP intent classification.
    """

    def __init__(self, replace_numbers: bool = True, lowercase: bool = True):
        self.replace_numbers = replace_numbers
        self.lowercase = lowercase

    def expand_contractions(self, text: str) -> str:
        """Expands common contractions in text (e.g. don't -> do not)."""
        def replace(match):
            key = match.group(0).lower()
            return CONTRACTION_MAP.get(key, match.group(0))
        return _CONTRACTION_PATTERN.sub(replace, text)

    def normalize_entities(self, text: str) -> str:
        """
        Replaces dynamic customer query entities (order numbers, currency amounts, emails)
        with generic tokens so the intent classifier focuses on intent verbs and semantics.
        """
        # Order / Tracking patterns (e.g., #12345, ORD-99881, #A12B)
        # 1. Any hash-prefixed identifier: #12345, #ORD12
        text = re.sub(r'#[A-Za-z0-9_-]+', ' <ENTITY_ID> ', text)
        # 2. Specific prefix tokens followed by separator or digits: ORD-12345, TRK_9981, REF123
        text = re.sub(r'\b(?:ord|order|trk|track|id|ref)(?:[-_][A-Za-z0-9]+|\d+)\b', ' <ENTITY_ID> ', text, flags=re.IGNORECASE)
        # 3. Alphanumeric codes containing both digits and letters (e.g. A98B12, 1234AB)
        text = re.sub(r'\b(?=[a-zA-Z]*[0-9])(?=[0-9]*[a-zA-Z])[a-zA-Z0-9_-]{4,}\b', ' <ENTITY_ID> ', text)

        # Currency amounts (e.g. $49.99, £20, €100)
        text = re.sub(r'[\$€£₹]\s*\d+(?:\.\d+)?', ' <AMOUNT> ', text)

        # Emails
        text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' <EMAIL> ', text)

        # Standalone numbers
        if self.replace_numbers:
            text = re.sub(r'\b\d+\b', ' <NUM> ', text)

        return text

    def clean_text(self, text: str) -> str:
        """
        Performs full end-to-end cleaning for a single query string.
        """
        if not isinstance(text, str):
            return ""

        # Step 1: Lowercase
        if self.lowercase:
            text = text.lower()

        # Step 2: Expand contractions
        text = self.expand_contractions(text)

        # Step 3: Entity normalization
        text = self.normalize_entities(text)

        # Step 4: Remove punctuation / special characters (keep alphanumeric, entity brackets, and underscores)
        text = re.sub(r'[^a-zA-Z0-9\s<>_]', ' ', text)

        # Step 5: Collapse multiple whitespaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def transform(self, texts: Union[str, List[str], pd.Series]) -> Union[str, List[str]]:
        """
        Batch or single text transformation.
        """
        if isinstance(texts, str):
            return self.clean_text(texts)
        if isinstance(texts, pd.Series):
            return texts.apply(self.clean_text).tolist()
        return [self.clean_text(t) for t in texts]
