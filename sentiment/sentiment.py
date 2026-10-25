import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Identify the model on Hugging Face
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# 2. Download and load the tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Download and load the trained model
print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

print("Model and tokenizer are ready!\n")

# 4. Enter the text you want to analyse
text = "I love learning about artificial intelligence!"

# 5. Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

print("Original text:")
print(text)

print("\nTokens:")
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

print("\nToken IDs:")
print(inputs["input_ids"][0].tolist())

print("\nAttention mask:")
print(inputs["attention_mask"][0].tolist())

# 6. Query the model (run inference)
with torch.no_grad():
    outputs = model(**inputs)

# 7. Get the raw model scores (logits)
logits = outputs.logits
print("\nRaw logits:")
print(logits)

# 8. Convert logits to probabilities
probabilities = torch.softmax(logits, dim=-1)[0]
print("\nProbabilities:")
print([f"{probability:.2%}" for probability in probabilities.tolist()])

# 9. Find the predicted sentiment
predicted_id = torch.argmax(probabilities).item()
sentiment = model.config.id2label[predicted_id]
confidence = probabilities[predicted_id].item()

print("\nSentiment result:")
print(f"Label: {sentiment}")
print(f"Score: {confidence:.4f}")
print(f"Percentage: {confidence * 100:.2f}%")
