CC=clang
CFLAGS=-Wall -Wextra -pedantic -std=c2x -Ofast
LIBS=-lm


INCLUDE_PATH=/usr/include/
LIB_PATH=/usr/lib64/

TARGET=linreg
CACHE=.cache
OUTPUT=$(CACHE)/release

MODULES += main.o


OBJ=$(addprefix $(CACHE)/,$(MODULES))


all: env $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) $(LIBS) -o $(OUTPUT)/$(TARGET)


%.o:
	$(CC) $(CFLAGS) -c $< -o $@


-include dep.list


exec: all $(T_OBJ)
	$(OUTPUT)/$(TARGET)


.PHONY: env dep clean


dep:
	$(CC) -MM app/*.c | sed 's|[a-zA-Z0-9_-]*\.o|$(CACHE)/&|' > dep.list


env:
	mkdir -pv $(CACHE)
	mkdir -pv $(OUTPUT)


clean: 
	rm -rvf $(OUTPUT)
	rm -vf $(CACHE)/*.o



