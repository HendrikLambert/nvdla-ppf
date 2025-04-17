
tensorflow:
	python3 tensorflow_model.py

run_pytorch:
	cd pytorch_model && make run_pytorch

compile_pytorch:
	cd pytorch_model && make compile_pytorch

tensorrt:
	python3 tensorrt_model.py

model: 
	cd runtime && make model

clean:
	cd pytorch_model && make clean
	cd runtime && make clean
