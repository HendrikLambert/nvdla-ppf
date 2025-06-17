# The model requires TensorRT >10.7 for the concatination layer to work properly.
TENSORRT := $(HOME)/TensorRT-10.7.0.23
TRTEXEC := $(TENSORRT)/bin/trtexec
export LD_LIBRARY_PATH=$(TENSORRT)/lib:$LD_LIBRARY_PATH

tensorflow:
	python3 tensorflow_model.py

run_pytorch:
	cd pytorch_model && python3 main.py

compile_pytorch:
	cd pytorch_model && $(TRTEXEC) --onnx=$(file) --verbose --fp16 --saveEngine=loadable.bin --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:chw16 --buildDLAStandalone --useDLACore=0

test_pytorch:
	cd pytorch_model && python3 -m unittest discover -s tests

tensorrt:
	python3 tensorrt_model.py

model: 
	cd runtime && make model

clean:
	# cd pytorch_model && make clean
	cd pytorch_model && rm *.dla || echo "no *.dla files to remove"
	cd pytorch_model && rm *.onnx || echo "no *.onnx files to remove"

	cd runtime && make clean

print_loadable:
	 python3 lb_reveng/lb_helper.py --loadable pytorch_model/loadable.bin

reference-model:
	cd reference/polyphase-filter-bank-generator && make
