from skl2onnx import to_onnx
import numpy as np
from onnxruntime import InferenceSession


def save_onnx(clf, X):
    onx = to_onnx(clf, X[:1].astype(np.float32), target_opset=12)
    with open("model/model.onnx", "wb") as f:
        f.write(onx.SerializeToString())


def load_onnx():
    with open("model/model.onnx", "rb") as f:
        onx = f.read()
    return onx


def classify_onnx(model, X):
    sess = InferenceSession(model, providers=["CPUExecutionProvider"])
    pred_ort = sess.run(None, {"X": X.astype(np.float32)})[0]
    return pred_ort
