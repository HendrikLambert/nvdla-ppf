CC = g++

CompileParms = -c -Wall -Wextra -Wpedantic -std=c++17 -O2 -I/usr/local/cuda/include
LinkerParms = -L/usr/local/cuda-12.6/targets/aarch64-linux/lib/ -lcudla -lcuda -lcudart

OBJS = main.o loadable.o benchmark.o cudla_dev.o cudla_runtime.o

model.o: model.cpp
	$(CC) $(CompileParms) model.cpp

loadable.o: loadable.cpp loadable.h
	$(CC) $(CompileParms) loadable.cpp

benchmark.o: benchmark.cpp benchmark.h
	$(CC) $(CompileParms) benchmark.cpp

cudla_dev.o: cudla_dev.cpp cudla_dev.h
	$(CC) $(CompileParms) cudla_dev.cpp

cudla_runtime.o: cudla_runtime.cpp cudla_runtime.h
	$(CC) $(CompileParms) cudla_runtime.cpp

main.o: main.cpp benchmark.h
	$(CC) $(CompileParms) main.cpp

benchmark: $(OBJS)
	$(CC) $(OBJS) $(LinkerParms) -o benchmark

clean:
	rm benchmark || echo "No benchmark to remove"
	rm *.o || echo "No object files to remove"
