import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertTokenizer
from transformers import pipeline

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from bnlp import NLTKTokenizer

# example = "জামাটা অনেক অনেক সুন্দর, আপনারা চাইলে নিতে পারেন"

csv_file_path = 'Sample_for_proposal.csv'
df = pd.read_csv(csv_file_path)

id = df['id'].values
review = df['review'].values
t_review = df['tagged_review'].values
sentiment = df['sentiment'].values

named_entity = [word.split('_NE_')[1] for review in t_review for word in review.split() if '_NE_' in word]

tokenizer = AutoTokenizer.from_pretrained("sagorsarker/mbert-bengali-ner")
model = AutoModelForTokenClassification.from_pretrained("sagorsarker/mbert-bengali-ner")
# nlp = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
# # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# # Use the tokenizer to encode the reviews
# encoded_data = tokenizer.batch_encode_plus(
#     list(review),
#     padding=True,
#     truncation=True,
#     return_tensors='pt'
# )

bnltk = NLTKTokenizer()
tokenized_review_List = []
tokenized_labels_List = []
tag = []
ner_tag_list = []

for SingleReview in review:
    # tokens = tokenizer(SingleReview, padding=True, truncation=True, return_tensors='pt')
    # ner_token = tokenizer(named_entity, padding=True, truncation=True, return_tensors='pt')
    sentence_tokens = bnltk.sentence_tokenize(text)

    word_Tokens = bnltk.word_tokenize(SingleReview)
    ner_token = bnltk.sentence_tokenize(SingleReview)
    tokenized_review = tokens
    tokenized_review_List.append(tokenized_review)
    tag = []
    for singleToken in tokens:
        if singleToken in ner_token:
            tag.append(1)
        else:
            tag.append(0)
    ner_tag_list.append(tag)
    print()


print(t_review)

# ner_results = nlp(example)
# print(ner_results)
