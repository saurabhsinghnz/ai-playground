from pathlib import Path

import torch
from quantize_model import load_quantized_model
from transformers import AutoTokenizer


def main() -> None:
    model_dir = Path(__file__).resolve().parent / "gpt2-sentiment-int8"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Quantized model directory not found: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = load_quantized_model(model_dir)

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


if __name__ == "__main__":
    main()
