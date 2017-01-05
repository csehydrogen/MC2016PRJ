# Copyright 2016 HeeHoon Kim
#
# Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
# http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
# <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
# option. This file may not be copied, modified, or distributed
# except according to those terms.

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
