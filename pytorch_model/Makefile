
run_pytorch:
	python3 torch_model.py

compile_pytorch:
	/usr/src/tensorrt/bin/trtexec --onnx=ppf.onnx --verbose --fp16 --saveEngine=loadable.bin --inputIOFormats=fp16:chw16 --outputIOFormats=fp16:dla_linear --buildDLAStandalone --useDLACore=0
	# /usr/src/tensorrt/bin/trtexec --onnx=ppf.onnx --verbose --fp16 --saveEngine=loadable.bin --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear --buildDLAStandalone --useDLACore=0

clean:
	rm *.dla || echo "no *.dla files to remove"
	rm *.onnx || echo "no *.onnx files to remove"
