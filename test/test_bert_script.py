import torch
from transformers import AutoTokenizer, BertModel

from onnxruntime.transformers import optimizer

model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bert")

model.eval()

text = "Hello, how are you?"
inputs = tokenizer(text, return_tensors="pt")
input_names = ["input_ids", "attention_mask"]
output_names = ["output"]

onnx_path = "bert_model.onnx"

torch.onnx.export(
    model,
    (inputs["input_ids"], inputs["attention_mask"]),
    onnx_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=20,
    verbose=True,
)
print(f"Model exported to {onnx_path}")

m = optimizer.optimize_model(
    onnx_path,
    model_type="bert",
    num_heads=0,
    hidden_size=0,
    opt_level=2,
    use_gpu="cpu",
    verbose=True,
)

m.save_model_to_file(
    "bert_model_optimized.onnx",
)
print(m.get_fused_operator_statistics())
