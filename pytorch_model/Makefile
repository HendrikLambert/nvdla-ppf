
run_pytorch:
	python3 torch_model.py
	# /usr/src/tensorrt/bin/trtexec --onnx=ppf.onnx --shapes=input:2x16x256 --saveEngine=model_gn.engine --exportProfile=model_gn.json --fp16 --useDLACore=0 --useSpinWait --separateProfileRun
	# /usr/src/tensorrt/bin/trtexec --onnx=ppf.onnx --shapes=input:265 --saveEngine=model_gn.engine --exportProfile=model_gn.json --fp16 --useDLACore=0 --useSpinWait --separateProfileRun --buildDLAStandalone
	# /usr/src/tensorrt/bin/trtexec --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear --onnx=ppf.onnx --buildDLAStandalone --fp16 --saveEngine=loadable2.bin --verbose

compile_pytorch:
	/usr/src/tensorrt/bin/trtexec --onnx=ppf.onnx --verbose --fp16 --saveEngine=loadable.bin --inputIOFormats=fp16:dla_linear --outputIOFormats=fp16:dla_linear --buildDLAStandalone --useDLACore=0

clean:
	rm *.dla || echo "no *.dla files to remove"
	rm *.onnx || echo "no *.onnx files to remove"
