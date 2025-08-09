import argparse

from flair.data import Sentence
from flair.models import SequenceTagger
from flask import Flask, jsonify, request


class Text2Entities:
    def __init__(self):
        self.tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

    def extract_entities(self, text: str, return_json=False):
        sentence = Sentence(text)
        self.tagger.predict(sentence)
        entities = sentence.get_spans("ner")
        formatted_entities = [
            {"word": entity.text, "tag": entity.tag} for entity in entities
        ]
        return formatted_entities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text to Entities Service")
    parser.add_argument(
        "--mode",
        choices=["test", "console", "API"],
        default="test",
        help="Mode to run the application: 'test', 'console', or 'API'",
    )
    args = parser.parse_args()
    text2entities = Text2Entities()

    if args.mode == "test":
        # input_text = "The capital of France is Paris."
        # input_text = "The founding time of hospitals?"
        input_text = "how many 3-point attempts did steve nash average per game in seasons he made the 50-40-90 club?"
        # input_text = "In which country did Tesla build a new Gigafactory in 2021?"
        extracted_entities = text2entities.extract_entities(input_text)
        print(f"Input Text: {input_text}")
        print(f"Extracted Entities: {extracted_entities}")
    elif args.mode == "console":
        # Terminal Interaction
        print("Welcome! (INPUT 'exit' TO EXIT)")
        while True:
            input_text = input("Query: ")
            if input_text.lower() == "exit":
                print("Exit.")
                break
            extracted_entities = text2entities.extract_entities(input_text)
            print(f"Extracted Entities: {extracted_entities}")
    elif args.mode == "API":
        app = Flask(__name__)

        @app.route("/extracted_entities", methods=["GET"])
        def generate_triple():
            input_text = request.args.get("query")
            if not input_text:
                return jsonify({"error": "Missing 'query' in request"}), 400

            extracted_entities = text2entities.extract_entities(
                text=input_text, return_json=True
            )
            return jsonify(extracted_entities)

        port = 20257
        print(f"Service is running on port {port}...")
        app.run(host="0.0.0.0", port=port)
    else:
        raise ValueError(
            f"Invalid mode {args.mode}. Please choose from ['test','console','API']."
        )
