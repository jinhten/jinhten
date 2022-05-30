#Start of the makefile

NAME=pbjson

CC = g++ -std=c++17

OBJDIR = obj
VPATH = ./src

OUTPUT_OPTION = -o $(OBJDIR)/$@

CXXFLAGS += -Wall -g -I./src \
			-I./src/rapidjson 


STATIC_LIB = $(OBJDIR)/lib$(NAME).a

OBJECTS := $(notdir $(patsubst %.cpp,%.o,$(wildcard src/*.cpp)))



all: $(STATIC_LIB)

$(STATIC_LIB): $(OBJECTS)
	ar rv $(STATIC_LIB) $(addprefix $(OBJDIR)/,$(OBJECTS))
#	$(CC) $(CXXFLAGS) -o $(STATIC_LIB) $(addprefix $(OBJDIR)/,$(OBJECTS))

.PHONY: clean
clean:
	rm -rf $(addprefix $(OBJDIR)/,$(OBJECTS)) $(STATIC_LIB)
