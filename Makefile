CC=g++
CFLAGS=-Wall -g
DEPS=opus.h randomlhs.hpp

SOURCE_FILES=$(shell find . -name '*.cpp')
OBJ_FILES=$(SOURCE_FILES:.c=.o) 
LIB=-lm 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
	
demo: $(OBJ_FILES)
	$(CC) $^ -o $@ $(LIB)


.PHONY: clean
clean:
	rm -f demo $(OBJ_FILES)
