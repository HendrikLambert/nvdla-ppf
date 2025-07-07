# The model requires TensorRT >10.7 for the concatination layer to work properly.
TENSORRT := $(HOME)/TensorRT-10.7.0.23
TRTEXEC := $(TENSORRT)/bin/trtexec
export LD_LIBRARY_PATH=$(TENSORRT)/lib:$LD_LIBRARY_PATH

# For scatch location
USER := $(shell whoami)
SCRATCH := /var/scratch/$(USER)

help:
	@echo "Available targets:"
	@echo "  create_onnx_benchmarkfiles - Create ONNX benchmark files"
	@echo "  create_nvdla_benchmarkfiles - Create NVDLA benchmark files"
	@echo "  run_pytorch - Run the PyTorch model"
	@echo "  compile_pytorch - Compile the PyTorch model to a loadable binary"
	@echo "  test_pytorch - Run unit tests for the PyTorch model"
	@echo "  model - Build the model in runtime"
	@echo "  build_benchmark - Build the benchmark in runtime"
	@echo "  clean - Clean up generated files"
	@echo "  print_loadable - Print information about a loadable file"
	@echo "  reference-model - Build the reference model"

create_onnx_benchmarkfiles:
	cd pytorch_model && python3 main.py --test --benchmark --type onnx --onnx_folder $(SCRATCH)/$(onnx_folder)

create_nvdla_benchmarkfiles:
	cd pytorch_model && python3 main.py --benchmark --type nvdla --onnx_folder $(SCRATCH)/$(onnx_folder) --nvdla_folder $(SCRATCH)/$(nvdla_folder) --trtexec_location $(TRTEXEC)

create_nvdla_custom:
	cd pytorch_model && python3 main.py --buildONNX --model_type pfb_dft -b 256 --onnx_file $(SCRATCH)/custom/custom_pfb_dft.onnx --buildNVDLA --loadable_location $(SCRATCH)/custom/custom_pfb_dft.nvdla --trtexec_location $(TRTEXEC) --verbose

run_pytorch:
	cd pytorch_model && python3 main.py

compile_pytorch:
	cd pytorch_model && $(TRTEXEC) --onnx=$(file) --verbose --fp16 --saveEngine=loadable.nvdla --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0

test_pytorch:
	cd pytorch_model && python3 -m unittest discover -s tests

build_simpleruntime: 
	cd runtime && make simple

build_benchmark:
	cd runtime && make benchmark

run_benchmark:
	./runtime/benchmark benchmark $(SCRATCH)/$(nvdla_folder) analysis/$(results)

clean:
	# cd pytorch_model && make clean
	cd pytorch_model && rm *.dla || echo "no *.dla files to remove"
	cd pytorch_model && rm *.onnx || echo "no *.onnx files to remove"

	cd runtime && make clean

print_loadable:
	 python3 lb_reveng/lb_helper.py --loadable $(file)

reference-model:
	cd reference/polyphase-filter-bank-generator && make
