CC = g++

# Power Sensor path
ps3 := $(HOME)/ps3

CompileParms = -c -Wall -Wextra -Wpedantic -std=c++17 -O2 -I/usr/local/cuda/include -I$(ps3)/include
LinkerParms = -L/usr/local/cuda-12.6/targets/aarch64-linux/lib/ -lcudla -lcuda -lcudart -L$(ps3)/lib -lPowerSensor

OBJS = main.o loadable.o benchmark.o cudla_runtime.o

model.o: model.cpp
	$(CC) $(CompileParms) model.cpp

model: model.o
	$(CC) model.o $(LinkerParms) -o model

loadable.o: loadable.cpp loadable.hpp
	$(CC) $(CompileParms) loadable.cpp

benchmark.o: benchmark.cpp benchmark.hpp
	$(CC) $(CompileParms) benchmark.cpp

cudla_runtime.o: cudla_runtime.cpp cudla_runtime.hpp
	$(CC) $(CompileParms) cudla_runtime.cpp

main.o: main.cpp
	$(CC) $(CompileParms) main.cpp

benchmark: $(OBJS)
	$(CC) $(OBJS) $(LinkerParms) -o benchmark

clean:
	rm benchmark || echo "No benchmark to remove"
	rm *.o || echo "No object files to remove"
