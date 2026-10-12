import json
import random
import re
import os
import sys
from datetime import datetime

import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK resources are downloaded quietly
for resource in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)


class AIChatbot:
    def __init__(self, intents_file="intents.json", log_file="session_chat_log.txt", threshold=0.28):
        self.intents_file = intents_file
        self.log_file = log_file
        self.threshold = threshold
        self.lemmatizer = WordNetLemmatizer()

        # Context-aware multi-turn state tracking
        self.context = {
            "current_topic": None,
            "user_name": None,
            "order_id": None,
            "turn_count": 0
        }

        # Load intents and train TF-IDF model
        self.intents_data = self._load_intents()
        self.corpus = []
        self.tags = []
        self._prepare_training_data()
        self.vectorizer = TfidfVectorizer(tokenizer=self.preprocess_text, token_pattern=None)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)

        # Initialize session log
        self._init_session_log()

    def _load_intents(self):
        if not os.path.exists(self.intents_file):
            raise FileNotFoundError(f"Intents file '{self.intents_file}' not found.")
        with open(self.intents_file, "r", encoding="utf-8") as file:
            return json.load(file)

    def preprocess_text(self, text):
        """Tokenize, clean, and lemmatize natural language input."""
        text = text.lower().strip()
        text = re.sub(r"[^a-z0-9\s]", "", text)
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def _prepare_training_data(self):
        for intent in self.intents_data["intents"]:
            for pattern in intent["patterns"]:
                self.corpus.append(pattern)
                self.tags.append(intent["tag"])

    def _init_session_log(self):
        with open(self.log_file, "a", encoding="utf-8") as log:
            log.write("\n" + "=" * 60 + "\n")
            log.write(f"Session Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write("=" * 60 + "\n")

    def log_interaction(self, user_input, bot_response, intent_tag, confidence):
        """Log conversation history for the current session."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a", encoding="utf-8") as log:
            log.write(f"[{timestamp}] USER: {user_input}\n")
            log.write(
                f"[{timestamp}] BOT ({intent_tag} | conf={confidence:.2f} | ctx={self.context['current_topic']}): {bot_response}\n"
            )

    def _handle_contextual_input(self, user_input):
        """
        Handle multi-turn context such as capturing user's name, Order ID,
        or resolving follow-up pronouns based on the previous turn.
        """
        # 1. Check if user introduces their name
        name_match = re.search(r"\b(?:my name is|i am|i'm|call me)\s+([a-zA-Z]+)\b", user_input, re.IGNORECASE)
        if name_match and name_match.group(1).lower() not in ["interested", "looking", "having", "asking", "here"]:
            self.context["user_name"] = name_match.group(1).capitalize()
            return (
                f"Nice to meet you, {self.context['user_name']}! How can I assist you with your orders or questions today?",
                "context_name_capture",
                1.0
            )

        # 2. Check if user provides a 6-digit Order ID during order_context
        order_match = re.search(r"#?(\d{6})\b", user_input)
        if order_match:
            self.context["order_id"] = order_match.group(1)
            self.context["current_topic"] = "order_context"
            name_prefix = f"{self.context['user_name']}, your" if self.context["user_name"] else "Your"
            return (
                f"Thank you! {name_prefix} Order #{self.context['order_id']} is currently 'In Transit' and is scheduled for delivery tomorrow by 6:00 PM.",
                "context_order_lookup",
                1.0
            )

        # 3. Handle short follow-up responses based on active context
        cleaned = user_input.lower().strip()
        if self.context["current_topic"] == "return_context" and cleaned in ["yes", "yes please", "sure", "tell me", "how long?"]:
            return (
                "Once your return is picked up, refunds are credited within 3 to 5 business days to your original payment source.",
                "refund_status",
                0.95
            )

        if self.context["current_topic"] == "order_context" and any(phrase in cleaned for phrase in ["cancel it", "modify it", "stop it"]):
            order_ref = f"Order #{self.context['order_id']}" if self.context["order_id"] else "your order"
            return (
                f"I have initiated a cancellation check for {order_ref}. Since it has not reached the final hub yet, you can confirm cancellation under 'My Orders' -> 'Cancel'.",
                "order_cancel",
                0.95
            )

        return None

    def predict_intent(self, user_input):
        """Recognize user intent using TF-IDF vectorization and Cosine Similarity."""
        input_vec = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(input_vec, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])
        predicted_tag = self.tags[best_idx]

        # Boost score slightly if the predicted intent matches the current conversation topic
        if self.context["current_topic"] == "order_context" and predicted_tag in ["order_cancel", "order_status"]:
            best_score = min(1.0, best_score + 0.10)
        elif self.context["current_topic"] == "return_context" and predicted_tag in ["refund_status", "return_policy"]:
            best_score = min(1.0, best_score + 0.10)

        if best_score < self.threshold:
            return "fallback", best_score

        return predicted_tag, best_score

    def get_response(self, user_input):
        """Generate a response for user input with context and fallback support."""
        self.context["turn_count"] += 1

        # Step 1: Check multi-turn context rules first
        contextual_result = self._handle_contextual_input(user_input)
        if contextual_result:
            response, tag, score = contextual_result
            self.log_interaction(user_input, response, tag, score)
            return response, tag, score

        # Step 2: Standard NLP intent classification
        intent_tag, confidence = self.predict_intent(user_input)

        if intent_tag == "fallback":
            fallback_responses = [
                "I'm sorry, I didn't quite understand that. Could you please rephrase? You can ask me about order tracking, returns, refunds, shipping, payment options, or discounts.",
                "I'm not sure I follow. Try asking about your order status, return policy, payment methods, or type 'help' to see what I can do!"
            ]
            response = random.choice(fallback_responses)
            self.log_interaction(user_input, response, "fallback", confidence)
            return response, "fallback", confidence

        # Step 3: Fetch intent response and update context
        for intent in self.intents_data["intents"]:
            if intent["tag"] == intent_tag:
                response = random.choice(intent["responses"])
                if intent.get("context_set") is not None and intent.get("context_set") != "":
                    self.context["current_topic"] = intent["context_set"]
                # Personalize response if user's name is known
                if self.context["user_name"] and intent_tag in ["greeting", "thanks", "goodbye"]:
                    response = f"{self.context['user_name']}, {response[0].lower() + response[1:]}"
                self.log_interaction(user_input, response, intent_tag, confidence)
                return response, intent_tag, confidence


def run_cli():
    """Run the interactive command-line chat interface."""
    bot = AIChatbot()
    print("=" * 65)
    print("         SHOPEASE AI CUSTOMER SUPPORT CHATBOT (CLI)         ")
    print("=" * 65)
    print("Commands: Type '/help' for supported topics | '/context' for state | 'exit' to quit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! Have a great day!")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit", "/quit"]:
            response, _, _ = bot.get_response("Goodbye")
            print(f"Bot: {response}")
            break

        if user_input.lower() == "/help":
            print(
                "\n[Supported Topics]:\n"
                " - Order Tracking & Status (e.g., 'Where is my order?')\n"
                " - Order Cancellation (e.g., 'Cancel my order')\n"
                " - Returns & Exchanges (e.g., 'What is your return policy?')\n"
                " - Refund Timelines (e.g., 'When will I get my refund?')\n"
                " - Payment Methods (e.g., 'Do you accept UPI?')\n"
                " - Shipping & Delivery (e.g., 'Do you offer free delivery?')\n"
                " - Discounts & Coupons (e.g., 'Are there any promo codes?')\n"
                " - Live Human Agent (e.g., 'Connect me to customer care')\n"
            )
            continue

        if user_input.lower() == "/context":
            print(f"\n[Current Session Context]: {bot.context}\n")
            continue

        response, tag, score = bot.get_response(user_input)
        print(f"Bot [{tag} | {score:.2f}]: {response}\n")


def run_test_suite():
    """Test the chatbot with 20 diverse user inputs (including multi-turn and fallback queries)."""
    bot = AIChatbot()
    test_inputs = [
        "Hello there!",
        "My name is Rahul",
        "Who are you and what can you do?",
        "Where is my order?",
        "My order ID is #458921",
        "Can I cancel it?",
        "What is your return policy?",
        "Yes please",
        "Where is my refund?",
        "What payment methods do you accept?",
        "Do you accept UPI?",
        "How much is shipping?",
        "Do you offer free delivery?",
        "Are there any promo codes or discounts today?",
        "I received a damaged item, how can I return a product?",
        "I want to talk to a human agent",
        "What is the weather on Mars right now?",          # Unknown query -> Fallback test 1
        "Can you write a recipe for chocolate cake?",     # Unknown query -> Fallback test 2
        "Thanks a lot for your help!",
        "Goodbye!"
    ]

    print("=" * 75)
    print("       RUNNING AUTOMATED TEST SUITE (20 DIVERSE USER INPUTS)       ")
    print("=" * 75)
    for idx, query in enumerate(test_inputs, 1):
        response, tag, score = bot.get_response(query)
        print(f"\nTest #{idx:02d}")
        print(f"  User : {query}")
        print(f"  Intent: {tag} (Confidence: {score:.2f} | Active Context: {bot.context['current_topic']})")
        print(f"  Bot  : {response}")
    print("\n" + "=" * 75)
    print(f"All 20 test interactions completed and logged to '{bot.log_file}'.")
    print("=" * 75)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_test_suite()
    else:
        run_cli()