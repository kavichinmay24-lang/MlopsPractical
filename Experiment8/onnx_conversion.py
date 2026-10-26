
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Save all generated files beside this Python script
PROJECT_DIR = Path(__file__).resolve().parent
ONNX_PATH = PROJECT_DIR / "model.onnx"


# Step 1: Define a simple PyTorch model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


def main():
    torch.manual_seed(42)

    # Step 2: Create the model in evaluation mode
    model = SimpleModel()
    model.eval()

    # Step 3: Create dummy input (batch size 1, 4 features)
    dummy_input = torch.randn(1, 4)

    print("PyTorch model created successfully.")
    print("Input shape:", tuple(dummy_input.shape))

    # Step 4: Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(ONNX_PATH),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False
    )

    print("\nModel exported to:", ONNX_PATH.name)

    # Step 5: Validate the ONNX model
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)

    print("ONNX model structure validated successfully.")

    # Step 6: Create an ONNX Runtime inference session
    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=["CPUExecutionProvider"]
    )

    # Step 7: Run inference in both frameworks
    with torch.no_grad():
        pytorch_output = model(dummy_input).cpu().numpy()

    onnx_output = session.run(
        ["output"],
        {"input": dummy_input.cpu().numpy()}
    )[0]

    # Step 8: Compare outputs
    print("\nPyTorch output:")
    print(pytorch_output)

    print("\nONNX Runtime output:")
    print(onnx_output)

    difference = np.max(np.abs(pytorch_output - onnx_output))
    print("\nMaximum absolute difference:", difference)

    if np.allclose(pytorch_output, onnx_output, rtol=1e-4, atol=1e-5):
        print("SUCCESS: PyTorch and ONNX outputs match.")
    else:
        raise RuntimeError("The PyTorch and ONNX outputs do not match.")


if __name__ == "__main__":
    main()
