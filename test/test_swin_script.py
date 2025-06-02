import requests
import torch
from PIL import Image
from transformers import AutoImageProcessor, SwinModel
from transformers.models.swin.modeling_swin import SwinAttention

from onnxruntime.transformers import optimizer

model = SwinModel.from_pretrained("hf-internal-testing/tiny-random-SwinModel")
model.eval()

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("hf-internal-testing/tiny-random-SwinModel")
inputs = image_processor(images=image, return_tensors="pt")

print(inputs)

input_names = ["pixel_values"]
output_names = ["output"]

onnx_path = "swin_model.onnx"

# see:
# https://github.com/huggingface/transformers/blob/1094dd34f73dae1d9a91a6632635934516612490/src/transformers/models/swin/modeling_swin.py#L552
torch.onnx.export(
    model,
    (inputs["pixel_values"],),
    onnx_path,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        "pixel_values": {0: "batch_size", 1: "sequence_length"},
        "output": {0: "batch_size", 1: "sequence_length"},
    },
    opset_version=20,
    verbose=True,
    export_modules_as_functions={SwinAttention},
)
print(f"Model exported to {onnx_path}")

m = optimizer.optimize_model(
    onnx_path,
    model_type="swin",
    num_heads=0,
    hidden_size=0,
    opt_level=2,
    use_gpu="cpu",
    verbose=True,
)

m.save_model_to_file(
    "swin_model_optimized.onnx",
)
print(m.get_fused_operator_statistics())
