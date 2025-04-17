CC = g++

CompileParms = -c -Wall -Wextra -Wpedantic -std=c++11 -O2 -I/usr/local/cuda/include
LinkerParms = -L/usr/local/cuda-12.6/targets/aarch64-linux/lib/ -lcudla -lcuda -lcudart

OBJS = model.o

model.o: model.cpp
	$(CC) $(CompileParms) model.cpp

model: $(OBJS)
	$(CC) $(OBJS) $(LinkerParms) -o model

clean:
	rm model || echo "No model to remove"
	rm *.o || echo "No object files to remove"
