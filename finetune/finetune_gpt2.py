import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


def main():
    model_name = "openai-community/gpt2"
    output_dir = "./gpt2-sentiment"
    epochs = 1
    batch_size = 16  # Increased batch size for faster training
    max_length = 64  # Reduced max length for faster training
    learning_rate = 5e-5

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Loading SST-2 and {model_name} on {device}...")

    dataset = load_dataset("sst2")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_batch(batch):
        return tokenizer(
            batch["sentence"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = dataset.map(tokenize_batch, batched=True)
    columns = ["input_ids", "attention_mask", "label"]
    tokenized.set_format(type="torch", columns=columns)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # Ensures dynamic padding for batches to speed up training
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=data_collator,
    )
    validation_loader = DataLoader(
        tokenized["validation"], batch_size=batch_size, collate_fn=data_collator
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            total_loss += outputs.loss.item()
            progress.set_postfix(loss=f"{outputs.loss.item():.4f}")

        model.eval()
        correct = 0
        example_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                labels = batch.pop("labels")
                predictions = model(**batch).logits.argmax(dim=-1)
                correct += (predictions == labels).sum().item()
                example_count += labels.size(0)

        average_loss = total_loss / len(train_loader)
        accuracy = correct / example_count
        print(
            f"Epoch {epoch + 1}: train loss={average_loss:.4f}, "
            f"validation accuracy={accuracy:.4f}"
        )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model and tokenizer saved to {output_dir}")


if __name__ == "__main__":
    main()