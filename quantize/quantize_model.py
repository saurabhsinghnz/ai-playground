"""Create a CPU int8 GPT-2 sentiment checkpoint and matching tokenizer.

Run from the repository root with ``python quantize/quantize_model.py``.
Use ``load_quantized_model(Path("quantize/gpt2-sentiment-int8"))`` to load it.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers.pytorch_utils import Conv1D


def replace_conv1d_with_linear(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            linear = nn.Linear(
                child.weight.shape[0],
                child.weight.shape[1],
                bias=child.bias is not None,
            )
            linear.weight = nn.Parameter(child.weight.detach().T.contiguous())
            if child.bias is not None:
                linear.bias = nn.Parameter(child.bias.detach().contiguous())
            setattr(module, name, linear)
        else:
            replace_conv1d_with_linear(child)


def set_quantized_engine() -> None:
    supported_engines = torch.backends.quantized.supported_engines
    quantized_engine = next(
        (
            engine
            for engine in ("qnnpack", "x86", "fbgemm")
            if engine in supported_engines
        ),
        None,
    )
    if quantized_engine is None:
        raise RuntimeError(
            f"No supported PyTorch quantization engine found: {supported_engines}"
        )
    torch.backends.quantized.engine = quantized_engine


def load_quantized_model(model_dir: Path) -> nn.Module:
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_config(config)
    replace_conv1d_with_linear(model)
    set_quantized_engine()
    model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    state_dict = torch.load(
        model_dir / "model-int8.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state_dict)
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a fine-tuned GPT-2 classifier to a CPU int8 TorchScript model."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("finetune/gpt2-sentiment"),
        help="Directory containing the fine-tuned Hugging Face model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quantize/gpt2-sentiment-int8"),
        help="Directory for the TorchScript model and tokenizer files.",
    )
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        parser.error(f"Model directory does not exist: {args.model_dir}")
    if args.output_dir.resolve() == args.model_dir.resolve():
        parser.error("Output directory must not be the source model directory.")

    config = AutoConfig.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_dir, config=config, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model.eval().to("cpu")
    replace_conv1d_with_linear(model)
    set_quantized_engine()
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    ).eval()

    example_inputs = tokenizer(
        ["A short example for tracing."],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.inference_mode():
        expected_logits = quantized_model(**example_inputs).logits

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), args.output_dir / "model-int8.pt")
    tokenizer.save_pretrained(args.output_dir)
    quantized_model.config.save_pretrained(args.output_dir)
    restored_model = load_quantized_model(args.output_dir)
    with torch.inference_mode():
        restored_logits = restored_model(**example_inputs).logits
    torch.testing.assert_close(restored_logits, expected_logits)
    print(f"Saved verified int8 model and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
