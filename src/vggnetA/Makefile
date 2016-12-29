CC=gcc
CFLAGS=-Wall -O2 -std=gnu99

LIBS=-lm -lrt -lOpenCL
LDFLAGS=${LIBS}

.PHONY: all vggnet clean

all: vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}

clean:
	rm -f vggnet.o main.o vggnet task*

run: vggnet
	thorq --add --device cpu ./vggnet image_list.txt

full: vggnet
	thorq --add --device cpu ./vggnet image_list_full.txt

profile: vggnet
	thorq --add --device cpu /opt/AMDAPP/AMDAPPProfiler-2.5/x86_64/sprofile -p -t -w . ./vggnet image_list.txt
