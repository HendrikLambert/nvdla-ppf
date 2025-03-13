CC		= g++
CC_FLAGS	= -g

CC_SOURCES	= FilterBank.cc FIR.cc PPF.cc Main.cc

CC_OBJECTS	= $(CC_SOURCES:%.cc=%.o)

CC_LINK_FLAGS	= -lfftw3f

%.o:		%.cc
		$(CC) $(CC_FLAGS) -c $< -o $@


polyphase-filter-bank-generator:	$(CC_OBJECTS)
		$(CC) $^ -o $@ $(CC_LINK_FLAGS)

plot: polyphase-filter-bank-generator
	./polyphase-filter-bank-generator 256 16 HAMMING 2> HAMMING-example.data
	./polyphase-filter-bank-generator 256 16 BLACKMAN 2> BLACKMAN-example.data
	./polyphase-filter-bank-generator 256 16 GAUSSIAN 2> GAUSSIAN-example.data
	./polyphase-filter-bank-generator 256 16 KAISER 2> KAISER-example.data
	gnuplot example.gnuplot > example.jpg

clean:
	rm -f *.o polyphase-filter-bank-generator *~ HAMMING-example.data BLACKMAN-example.data GAUSSIAN-example.data KAISER-example.data example.pdf
