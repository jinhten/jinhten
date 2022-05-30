#Start of the makefile

NAME=zibnetproto

CC = g++ -std=c++17

OBJDIR = obj
PROTODIR = proto
VPATH = ./src

#OUTPUT_OPTION = -o $(OBJDIR)/$@
#OUTPUT_OPTION = -o $(PROTODIR)/$@

CXXFLAGS += -Wall -g -I./proto -I./ \



#OBJECTS := $(notdir $(patsubst %.cc,%.o,$(wildcard src/*.cc)))
OBJECTS = $(notdir $(patsubst %.cc,%.o,$(wildcard proto/*.cc)))


PROTO_FILES += FileInfo.proto \

STATIC_LIB = $(OBJDIR)/lib$(NAME).a

all: $(STATIC_LIB)

FileInfo.pb.o: proto/FileInfo.pb.cc
	$(CC) $(CXXFLAGS) -c proto/FileInfo.pb.cc

.PHONY: proto
proto: $(OBJECTS)
	protoc --cpp_out=. $(addprefix $(PROTODIR)/,$(PROTO_FILES))
	#rm -rf src hdr
	#mkdir src && mv $(PROTODIR)/*.cc src
	#mkdir hdr && mv $(PROTODIR)/*.h hdr


$(STATIC_LIB): $(OBJECTS)
	#protoc --cpp_out=. $(addprefix $(PROTODIR)/,$(PROTO_FILES))
	#$(CC) $(CXXFLAGS) -c $(addprefix $(PROTODIR)/,$(wildcard *.pb.cc))
	mv *pb.o $(OBJDIR)
	ar rv $(STATIC_LIB) $(addprefix $(OBJDIR)/,$(OBJECTS))

.PHONY: clean
clean:
	#rm -rf $(addprefix $(OBJDIR)/,$(OBJECTS)) $(STATIC_LIB)
	rm -rf obj/* $(PROTODIR)/*pb*
