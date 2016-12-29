CC=mpicc
CFLAGS=-Wall -O2 -std=gnu99

LDFLAGS=-lm -lrt -L$(SNUCLROOT)/lib -lsnucl_cluster -pthread

.PHONY: all vggnet clean

all: vggnet

vggnet: vggnet.o main.o
	${CC} $^ -o $@ ${LDFLAGS}

info: opencl_info
	thorq --add --mode snucl --nodes 4 --device gpu ./opencl_info

clean:
	rm -f *.o vggnet task*

run: vggnet
	thorq --add --mode snucl --nodes 4 --device gpu ./vggnet image_list.txt

full: vggnet
	thorq --add --mode snucl --nodes 4 --device gpu ./vggnet image_list_full.txt

perf: vggnet
	thorq --add --device gpu /opt/AMDAPP/AMDAPPProfiler-2.5/x86_64/sprofile -p -w . ./vggnet image_list.txt

trace: vggnet
	thorq --add --device gpu /opt/AMDAPP/AMDAPPProfiler-2.5/x86_64/sprofile -t -w . ./vggnet image_list.txt

summary:
	/opt/AMDAPP/AMDAPPProfiler-2.5/x86_64/sprofile -T -a ~/cltrace.atp
	mv ~/*.html ~/html

