# The model requires TensorRT >10.7 for the concatination layer to work properly.
TENSORRT := $(HOME)/TensorRT-10.7.0.23
TRTEXEC := $(TENSORRT)/bin/trtexec
export LD_LIBRARY_PATH=$(TENSORRT)/lib:$LD_LIBRARY_PATH

# For scatch location
USER := $(shell whoami)

create_onnx_benchmarkfiles:
	cd pytorch_model && python3 main.py --test --benchmark --type onnx --onnx_folder /var/scratch/$(USER)/$(onnx_dir)

create_nvdla_benchmarkfiles:
	cd pytorch_model && python3 main.py --benchmark --type nvdla --onnx_folder /var/scratch/$(USER)/$(onnx_dir) --nvdla_folder /var/scratch/$(USER)/$(nvdla_dir) --trtexec_location $(TRTEXEC) 

run_pytorch:
	cd pytorch_model && python3 main.py

compile_pytorch:
	cd pytorch_model && $(TRTEXEC) --onnx=$(file) --verbose --fp16 --saveEngine=loadable.bin --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0

test_pytorch:
	cd pytorch_model && python3 -m unittest discover -s tests

model: 
	cd runtime && make model

clean:
	# cd pytorch_model && make clean
	cd pytorch_model && rm *.dla || echo "no *.dla files to remove"
	cd pytorch_model && rm *.onnx || echo "no *.onnx files to remove"

	cd runtime && make clean

print_loadable:
	 python3 lb_reveng/lb_helper.py --loadable $(file)

reference-model:
	cd reference/polyphase-filter-bank-generator && make
