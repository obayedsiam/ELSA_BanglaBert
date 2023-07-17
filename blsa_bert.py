import torch
import random
from sklearn.metrics import precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

# Set a random seed for reproducibility
random.seed(42)

# Load the pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Define the dataset
dataset = [
    "I absolutely loved the _NE_iPhone! The camera quality is _POS_outstanding.",
    "The _NE_service at this restaurant was _NEG_horrible, but the _NE_food was _POS_fantastic.",
    "The _NE_movie was _NEG_disappointing. The plot lacked depth.",
    "I had a great experience with the _NE_customer support of this company.",
    "The _NE_new car I purchased is a dream. I'm extremely satisfied.",
    "_NE_Mount Everest is the highest peak in the world.",
    "The _NE_book I read last week was quite interesting.",
    "_NE_Starbucks is my go-to coffee shop.",
    "The _NE_cat I adopted is adorable.",
    "_NE_London is a vibrant city with rich history.",
    "I enjoyed the _NE_concert I attended yesterday.",
    "The _NE_laptop I bought works well.",
    "I visited _NE_Paris last summer.",
    "The _NE_painting in the art gallery caught my attention.",
    "I had lunch at the _NE_cafe near my workplace."
]

# Define labels for sentiment classes
labels = ["_NEG_", "_POS_"]

# Split the dataset into training, testing, and validation sets
train_data, test_val_data = train_test_split(dataset, test_size=0.3, random_state=42)
test_data, val_data = train_test_split(test_val_data, test_size=0.5, random_state=42)


# Define function for sentiment analysis
def predicted_sentiment(sentences):
    input_ids = []
    labels2 = []

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        input_ids.append(tokenizer.convert_tokens_to_ids(tokens))

        # Get the sentiment label from the sentence
        sentiment_label = "_POS_" if "_POS_" in sentence else "_NEG_"
        labels2.append(sentiment_label)

    # Pad input sequences
    max_len = max(len(ids) for ids in input_ids)
    padded_input_ids = [ids + [0] * (max_len - len(ids)) for ids in input_ids]

    # Convert lists to tensors
    input_ids = torch.tensor(padded_input_ids)
    labels2 = torch.tensor(labels2)

    # Perform sentiment analysis
    outputs = model(input_ids)[0]
    predicted_labels = torch.argmax(outputs, dim=1)

    return labels2, predicted_labels


# Perform sentiment analysis on training set
train_labels, train_predictions = predicted_sentiment(train_data)

# Perform sentiment analysis on testing set
test_labels, test_predictions = predicted_sentiment(test_data)

# Perform sentiment analysis on validation set
val_labels, val_predictions = predicted_sentiment(val_data)

# Calculate evaluation metrics
train_precision = precision_score(train_labels, train_predictions, average='weighted')
test_precision = precision_score(test_labels, test_predictions, average='weighted')
val_precision = precision_score(val_labels, val_predictions, average='weighted')

train_accuracy = accuracy_score(train_labels, train_predictions)
test_accuracy = accuracy_score(test_labels, test_predictions)
val_accuracy = accuracy_score(val_labels, val_predictions)

train_f1 = f1_score(train_labels, train_predictions, average='weighted')
test_f1 = f1_score(test_labels, test_predictions, average='weighted')
val_f1 = f1_score(val_labels, val_predictions, average='weighted')

# Print evaluation metrics
print("Training Precision:", train_precision)
print("Testing Precision:", test_precision)
print("Validation Precision:", val_precision)
print("------------")
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Validation Accuracy:", val_accuracy)
print("------------")
print("Training F1 Score:", train_f1)
print("Testing F1 Score:", test_f1)
print("Validation F1 Score:", val_f1)
