#Start of the makefile

NAME=filefirst


CC = g++ #-std=c++17

OBJDIR = obj
VPATH = ./src

OUTPUT_OPTION = -o $(OBJDIR)/$@

DEFINE   += -D_FILE_OFFSET_BITS=64 #-D__CLI__#-fshort-wchar

# default heap, stack 1MB = 1048576
# check stack : ulimit -a
# change stack : ulimit -s [size]
OPTION   += -Wall -g -Wl,--no-keep-memory,--reduce-memory-overheads \
			-static-libstdc++ #-heap=4294967296 

#CXXFLAGS += -std=c++1z -Wall -g \

CXXFLAGS += -std=c++20 \
			$(DEFINE) \
			$(OPTION) \
			-I./hdr \


LIB_DIRS +=  \
			-L/usr/local/lib \

LIBS += -Xlinker --start-group \
		-lpthread \
		-Xlinker --end-group
		#-lpbjson \
		-lprotobuf \
		-lprotoc \


TARGET = $(OBJDIR)/lib$(NAME).a

OBJECTS = $(notdir $(patsubst %.cpp,%.o,$(wildcard src/*.cpp)))



all: $(TARGET)

$(TARGET): $(OBJECTS)
	ar rv $(TARGET) $(addprefix $(OBJDIR)/,$(OBJECTS))
	#$(CC) $(CXXFLAGS) -o $(TARGET) $(addprefix $(OBJDIR)/,$(OBJECTS)) $(LIB_DIRS) $(LIBS)

.PHONY: clean
clean:
	rm -rf $(addprefix $(OBJDIR)/,$(OBJECTS)) $(TARGET)
