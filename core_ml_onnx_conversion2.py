import coremltools
import onnx_coreml

ONNX_PATH = "./onnx_models/classification/espnetv2/espnetv2_s_2.0_imsize_224x224_imagenet.onnx"
# onnx_model = coremltools.utils.load_spec(ONNX_PATH)

# Convert the ONNX model to a CoreML model
mlmodel = onnx_coreml.convert(model=ONNX_PATH)
mlmodel.save("espnetv2_s_2.0_imsize_224x224_imagenet.mlmodel")