CC=gcc
CFLAGS=-O3
CLIBS=-lpthread -lnuma -lm -mavx -lpfm

all:	lsb_32 cmp_32 msb_32 chiplet_lsb_64 cmp_64 msb_64 lsb_64 lsb_64_radix_bits_64 chiplet_lsb_32 chiplet_cmp_64

lsb_32:	lsb_32.c init.c rand.c zipf.c shuffle.c
	${CC} ${CFLAGS} -o lsb_32 lsb_32.c rand.c init.c zipf.c shuffle.c ${CLIBS}

msb_32: msb_32.c init.c rand.c zipf.c shuffle.c
	${CC} ${CFLAGS} -o msb_32 msb_32.c rand.c init.c zipf.c shuffle.c ${CLIBS}

cmp_32: cmp_32.c init.c rand.c zipf.c shuffle.c
	${CC} ${CFLAGS} -o cmp_32 cmp_32.c rand.c init.c zipf.c shuffle.c ${CLIBS}

lsb_64: lsb_64.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o lsb_64 lsb_64.c rand.c init.c zipf.c ${CLIBS}

chiplet_cmp_64: cmp_64_chiplet.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o chiplet_cmp_64 cmp_64_chiplet.c rand.c init.c zipf.c ${CLIBS}

chiplet_lsb_64: lsb_64_chiplet.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o chiplet_lsb_64 lsb_64_chiplet.c rand.c init.c zipf.c ${CLIBS}

chiplet_lsb_32: lsb_32_chiplet.c init.c rand.c zipf.c shuffle.c
	${CC} ${CFLAGS} -o chiplet_lsb_32 lsb_32_chiplet.c rand.c init.c zipf.c shuffle.c ${CLIBS}

lsb_64_radix_bits_64: lsb_64_radix_bits_64.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o lsb_64_radix_bits_64 lsb_64_radix_bits_64.c rand.c init.c zipf.c ${CLIBS}

msb_64: msb_64.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o msb_64 msb_64.c rand.c init.c zipf.c ${CLIBS}

cmp_64: cmp_64.c init.c rand.c zipf.c
	${CC} ${CFLAGS} -o cmp_64 cmp_64.c rand.c init.c zipf.c ${CLIBS}

clean:
	#rm -f lsb_32 msb_32 cmp_32 lsb_64 msb_64 cmp_64 msb_64_8_threads
	rm -f lsb_32 cmp_32 msb_32 chiplet_lsb_64 cmp_64 msb_64 lsb_64 lsb_64_radix_bits_64 chiplet_lsb_32 chiplet_cmp_64
