from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
model = AutoModelForSequenceClassification.from_pretrained("sagorsarker/bangla-bert-base")


# Define the data with multiple entities and their corresponding sentiments
data = [
    {
        "text": "আমার বাংলাদেশ অসাধারণ একটি দেশ। ঢাকা শহরটি আমার খুবই পছন্দ।",
        "entities": [
            {"entity": "বাংলাদেশ", "sentiment": "positive"},
            {"entity": "ঢাকা", "sentiment": "negative"}
        ]
    },
    {
        "text": "এখানে খুব সুন্দর পরিবেশ আছে। মানুষের সবার সাথেই আদরের ব্যবহার হচ্ছে।",
        "entities": [
            {"entity": "পরিবেশ", "sentiment": "positive"},
            {"entity": "মানুষ", "sentiment": "positive"}
        ]
    },
    # Add more data with sentences, entities, and sentiments
]


# Function to preprocess text
def preprocess_text(text):
    return text  # Add any preprocessing steps if required


# Function to predict sentiment for a given text
def predict_sentiment(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128,
    )
    input_ids = inputs["input_ids"].to(torch.device("cpu"))
    attention_mask = inputs["attention_mask"].to(torch.device("cpu"))

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
    return predicted_labels[0]


# Perform entity-level sentiment analysis
results = []
for item in data:
    text = preprocess_text(item["text"])
    entities = item["entities"]
    entity_results = []

    for entity in entities:
        entity_text = entity["entity"]
        entity_sentiment = entity["sentiment"]
        predicted_sentiment = predict_sentiment(entity_text)

        entity_result = {
            "entity": entity_text,
            "sentiment": entity_sentiment,
            "predicted_sentiment": predicted_sentiment
        }
        entity_results.append(entity_result)

    result = {
        "text": text,
        "entity_results": entity_results
    }
    results.append(result)

# Print the results
for result in results:
    print("Text:", result["text"])
    print("Entity-Level Sentiment Analysis:")
    for entity_result in result["entity_results"]:
        print("Entity:", entity_result["entity"])
        print("Expected Sentiment:", entity_result["sentiment"])
        print("Predicted Sentiment:", entity_result["predicted_sentiment"])
        print()
    print()
