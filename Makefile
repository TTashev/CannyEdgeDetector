CC = g++
CFLAGS = -Wall -std=c++11
LDFLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

TARGET = imageFeatureDetection
SRC = imageFeatureDetection.cpp

all: $(TARGET)

$(TARGET): $(SRC) 
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
