CC = clang++
CFLAGS = -std=c++11 -O3 -Wall -Wextra -Werror -Weffc++ -Wstrict-aliasing --pedantic

.PHONY : all
all : csvtrain difftrain jitter

csvtrain : csvtrain.cpp mixmodel.cpp
	$(CC) $(CFLAGS) csvtrain.cpp mixmodel.cpp -o csvtrain

difftrain : difftrain.cpp mixmodel.cpp
	$(CC) $(CFLAGS) difftrain.cpp mixmodel.cpp -o difftrain

jitter : jitter.cpp mixmodel.cpp
	$(CC) $(CFLAGS) jitter.cpp mixmodel.cpp -o jitter

.PHONY : clean
clean :
	rm -f csvtrain difftrain jitter &> /dev/null
