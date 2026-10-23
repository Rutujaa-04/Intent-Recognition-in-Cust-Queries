"""
Inference CLI and Interactive REPL for Customer Intent Recognition.
Usage:
    # Single query prediction:
    python predict.py --text "Where is my package #9812?"

    # Interactive REPL mode:
    python predict.py
"""
import argparse
import sys
from src.models.intent_classifier import CustomerIntentClassifier
from src import config


def format_prediction_output(result: dict) -> None:
    print("\n" + "=" * 55)
    print(f" Query:        \"{result['query']}\"")
    print(f" Normalized:   \"{result['cleaned_query']}\"")
    print("-" * 55)

    intent_display = result["intent"]
    if result["requires_human_escalation"]:
        print(f" PREDICTED INTENT: \033[93m{intent_display}\033[0m (LOW CONFIDENCE)")
        print(" ACTION:           ⚠️  Route to Human Support Agent")
    else:
        print(f" PREDICTED INTENT: \033[92m{intent_display}\033[0m")
        print(" ACTION:           🤖 Automated Intent Fulfillment")

    print(f" Confidence:       {result['confidence'] * 100:.2f}%")

    if result.get("explanation"):
        print(f" Key Tokens:       {', '.join(result['explanation'])}")

    print("\n Top Candidates:")
    for rank, candidate in enumerate(result.get("top_intents", []), 1):
        bar_len = int(candidate["probability"] * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"   {rank}. {candidate['intent']:<26} [{bar}] {candidate['probability']*100:5.1f}%")
    print("=" * 55 + "\n")


def interactive_mode(classifier: CustomerIntentClassifier) -> None:
    print("\n" + "=" * 60)
    print("  Customer Query Intent Classifier - Interactive Console")
    print("  Type your customer query and press Enter.")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Customer Query > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting interactive console. Goodbye!")
                break
            result = classifier.predict(query)
            format_prediction_output(result)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting interactive console.")
            break


def main():
    parser = argparse.ArgumentParser(description="Customer Query Intent Classifier")
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single customer query to classify"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.CONFIDENCE_THRESHOLD,
        help=f"Confidence threshold for fallback routing (default: {config.CONFIDENCE_THRESHOLD})"
    )

    args = parser.parse_args()

    try:
        classifier = CustomerIntentClassifier(confidence_threshold=args.threshold)
    except FileNotFoundError:
        print("Model weights not found! Please run 'python train.py' first.")
        sys.exit(1)

    if args.text:
        result = classifier.predict(args.text)
        format_prediction_output(result)
    else:
        interactive_mode(classifier)


if __name__ == "__main__":
    main()
