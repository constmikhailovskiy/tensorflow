import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph("/tmp/onereach/saved_model.pb", input_arrays=['decoded_sample_data', 'decoded_sample_data:1'], output_arrays=['labels_softmax'])
converter.allow_custom_ops=True
tflite_model = converter.convert()
open("/tmp/onereach/converted_model.tflite", "wb").write(tflite_model)