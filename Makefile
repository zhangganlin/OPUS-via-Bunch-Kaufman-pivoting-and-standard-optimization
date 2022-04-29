CC=g++
CFLAGS=-Wall -g
DEPS=opus.h randomlhs.hpp

SOURCE_FILES=$(./opus.cpp ./demo.cpp ./randomlhs.cpp)
OBJ_FILES=$(SOURCE_FILES:.cpp=.o) 
LIB=-lm 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)
	
demo: $(OBJ_FILES)
	$(CC) $^ -o $@ $(LIB)
	rm -f ${OBJ_FILES}

.PHONY: clean
clean:
	rm -f demo $(OBJ_FILES)
