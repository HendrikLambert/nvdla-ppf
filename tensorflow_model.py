import tensorflow as tf
import numpy as np
import tf2onnx

class PPFModel(tf.keras.Model):
    def __init__(self):
        super(PPFModel, self).__init__()
        self.cnn = tf.keras.layers.Conv3D(filters=10, kernel_size=(1, 16, 32))
        # self.cnn.wei
        # print(self.cnn.weights[0].shape)

    def call(self, inputs):
        x = self.cnn(inputs)
        return x

def main():
    ppf_model = PPFModel()
    example_inputs = tf.random.normal((10, 16, 32, 1))

    out = ppf_model(example_inputs)

    # Convert to ONNX
    spec = (tf.TensorSpec((None, 16, 32, 1), tf.float32, name="input"),)
    output_path = "ppf.onnx"
    model_proto, _ = tf2onnx.convert.from_function(ppf_model.call, input_signature=spec, output_path=output_path)
    
    # Print the ONNX model
    print(model_proto)

if __name__ == "__main__":
    main()
