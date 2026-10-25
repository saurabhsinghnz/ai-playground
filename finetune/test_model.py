import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_dir = "./gpt2-sentiment"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSequenceClassification.from_pretrained(model_dir)
model.eval()

sentences = [
    "This movie was fantastic. I really enjoyed it!",
    "The story was boring and the acting was terrible.",
    "It started slowly, but the ending was wonderful.",
]

inputs = tokenizer(
    sentences,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
)

with torch.inference_mode():
    logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)

for sentence, scores in zip(sentences, probabilities):
    predicted_id = scores.argmax().item()
    label = model.config.id2label[predicted_id]
    score = scores[predicted_id].item()

    print(f"Text: {sentence}")
    print(f"Prediction: {label} | Score: {score:.1%}")
    print()