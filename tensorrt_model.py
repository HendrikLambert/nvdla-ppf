import tensorrt as trt

class Settings(object):
    INPUT_NAME = "data_in"
    WEIGHTS_NAME = "weights"
    INPUT_SHAPE = (1, 1, 256, 64)
    WEIGHTS_SHAPE = (1, 1, 256, 64)
    OUTPUT_NAME = "data_out"
    FIR_NAME = "fir_weights"
    FIR_SHAPE = INPUT_SHAPE


def main():
    # create logger
    logger = trt.Logger(trt.Logger.INFO)
    # create builder
    builder = trt.Builder(logger)


    # Network
    network = builder.create_network(
        trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED
        # 0
        )

    # Specify input
    input_tensor = network.add_input(Settings.INPUT_NAME, trt.DataType.HALF, shape=trt.Dims(Settings.INPUT_SHAPE))
    weight_tensor = network.add_input(Settings.WEIGHTS_NAME, trt.DataType.HALF, shape=trt.Dims(Settings.WEIGHTS_SHAPE))
    # input_tensor.allowed_formats = 1 << int(trt.TensorFormat.DLA_LINEAR)
    # print(input_tensor.is_shape)
    # fir_tensor = network.add_input(Settings.FIR_NAME, trt.DataType.HALF, shape=trt.Dims(Settings.FIR_SHAPE))


    # Add element wise (FIR multiplication)
    fir_mul_tensor = network.add_elementwise(input_tensor, weight_tensor, trt.ElementWiseOperation.SUM)

    # relu1 = network.add_activation(input=input_tensor, type=trt.ActivationType.RELU)
    
    out = fir_mul_tensor.get_output(0)
    out.allowed_formats = (1 << int(trt.TensorFormat.DLA_LINEAR))
    assert out.allowed_formats == (1 << int(trt.TensorFormat.DLA_LINEAR)), "Output tensor does not have DLA_LINEAR format"
    
    
    print("formats:", fir_mul_tensor.get_output(0).allowed_formats, trt.TensorFormat.DLA_LINEAR, 1 << int(trt.TensorFormat.DLA_LINEAR), out.allowed_formats)

    network.mark_output(out)


    # Builder
    config = builder.create_builder_config()
    config.default_device_type = trt.DeviceType.DLA
    config.engine_capability = trt.EngineCapability.DLA_STANDALONE
    config.DLA_core = 0
    # config.set_flag(trt.BuilderFlag.DIRECT_IO)
    config.set_flag(trt.BuilderFlag.FP16)

    serialized_engine = builder.build_serialized_network(network, config)
    # with open("loadable2.dla", "wb") as f:
    #     f.write(serialized_engine)


if __name__ == "__main__":
    main()