CC=g++
CFLAGS=-Wall -g
DEPS=opus.h

SOURCE_FILES=$(shell find . -name '*.c')
OBJ_FILES=$(SOURCE_FILES:.c=.o) 
LIB=-lm 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
	
demo: $(OBJ_FILES)
	$(CC) $^ -o $@ $(LIB)


.PHONY: clean
clean:
	rm -f demo $(OBJ_FILES)
