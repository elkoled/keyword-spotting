import os
import tensorflow as tf

MAIN_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(MAIN_DIR, "models", "svdf", "stream_state_internal")
OUTPUT_PATH = os.path.join(MODEL_DIR, "model.tflite")

# Initialize the TFLiteConverter to load the SavedModel
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_DIR)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]

# Set the optimizations
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Optionally, you can also specify a representative dataset for optimization
# def representative_dataset():
#     for _ in range(100):
#         data = np.random.rand(1, 224, 224, 3)
#         yield [data.astype(np.float32)]
#
# converter.representative_dataset = representative_dataset

# Convert the model
tflite_model = converter.convert()

# Save the TF Lite model
with tf.io.gfile.GFile(OUTPUT_PATH, 'wb') as f:
    f.write(tflite_model)
