import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info
import os

print("\n==== TENSORFLOW GPU DEBUG ====\n")

# TensorFlow info
print("TF Version:", tf.__version__)
print("TF Built With CUDA:", tf.test.is_built_with_cuda())
print("CUDA Version TF Was Built For:", tf_build_info.cuda_version_number)
print("cuDNN Version TF Was Built For:", tf_build_info.cudnn_version_number)

# GPU detection
print("\nDetected Physical GPUs:")
gpus = tf.config.list_physical_devices("GPU")
print(gpus)

# Environment CUDA variables
print("\nEnvironment CUDA Paths:")
print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
print("PATH entries containing CUDA:")
for p in os.environ["PATH"].split(";"):
    if "CUDA" in p or "cudnn" in p.lower():
        print("  -", p)

# Try GPU operation
print("\nTrying simple GPU matmul...")
try:
    with tf.device("/GPU:0"):
        a = tf.random.uniform((500, 500))
        b = tf.random.uniform((500, 500))
        c = tf.matmul(a, b)
    print("GPU MatMul succeeded! GPU is working.")
except Exception as e:
    print("GPU MatMul FAILED:", e)

print("\n==== END DEBUG ====\n")
