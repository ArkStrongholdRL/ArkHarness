"""Export the PP-OCRv6 recognizer with a digit-only CTC alphabet for MaaFramework."""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
from onnx import TensorProto, helper


def main():
    root = Path(__file__).resolve().parents[1] / "resource/model/ocr"
    keys = (root / "keys.txt").read_text(encoding="utf-8-sig").splitlines()
    model = onnx.load(root / "rec.onnx")
    output, = model.graph.output
    dimensions = output.type.tensor_type.shape.dim
    if len(dimensions) != 3 or dimensions[-1].dim_value != len(keys) + 2:
        raise ValueError("Expected PP-OCR CTC scores for blank, dictionary and space")

    digits = "0123456789"
    indices = [0, *(keys.index(digit) + 1 for digit in digits), len(keys) + 1]
    # Keep the original probability scale so non-digit glyphs can still fail the OCR confidence threshold.
    model.graph.initializer.append(helper.make_tensor("digit_indices", TensorProto.INT64, [len(indices)], indices))
    model.graph.node.append(helper.make_node("Gather", [output.name, "digit_indices"], ["digit_scores"], axis=2))
    output.name = "digit_scores"
    dimensions[-1].dim_value = len(indices)
    onnx.checker.check_model(model)

    destination = root / "digits"
    destination.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=root) as folder:
        pending = Path(folder)
        onnx.save(model, pending / "rec.onnx")
        (pending / "keys.txt").write_text("\n".join(digits) + "\n", encoding="utf-8")
        shutil.copyfile(root / "det.onnx", pending / "det.onnx")
        for path in pending.iterdir():
            path.replace(destination / path.name)


if __name__ == "__main__":
    main()
