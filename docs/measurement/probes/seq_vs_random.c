// seqrand — does a layer-major prefill's whole-file read beat the engine's scattered
// per-expert reads on THIS drive?
//
// Path 5 of the batched-prefill work. Layer-major reads ~246 of a layer's 257 experts, so
// it can read the whole L{ll}.vq3 file instead of scattering 246 random O_DIRECT reads
// through it. The question is purely what the DEVICE does with the two offset orders.
//
// Controlled: same request size, same request count, same total bytes, same queue depth.
// ONLY the offset order differs — `rand` picks aligned expert slots at random, `seq` walks
// them in ascending order (round-robin across threads, so the aggregate LBA stream is
// ascending rather than QD independent streams). That isolates sequentiality from request
// size, which a "big sequential read vs small random reads" comparison would confound.
//
// `-chunk` then asks the second question separately: does a LARGER request help, given
// layer-major is free to read in any granularity it likes?
//
// Threads, not io_uring: one blocking pread per outstanding read expresses queue depth
// without introducing a second variable — same choice, for the same reason, as
// probes/fetch_batch.hip.
#define _GNU_SOURCE
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define EXPERT 15335424L // one gate|up|down block, VQ_ALIGN=4096 aligned

static long chunk = EXPERT;
static int nreq, qd, seq;
static const char *path;
static long *offs;

static void *worker(void *a) {
    int id = (int)(long)a;
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); exit(1); }
    void *buf;
    if (posix_memalign(&buf, 4096, chunk)) { perror("memalign"); exit(1); }
    // Round-robin so the union of threads walks offsets in ascending order in `seq`.
    for (int i = id; i < nreq; i += qd) {
        long got = 0;
        while (got < chunk) {
            long r = pread(fd, (char *)buf + got, chunk - got, offs[i] + got);
            if (r <= 0) { if (got) break; perror("pread"); exit(1); }
            got += r;
        }
    }
    free(buf); close(fd);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: seqrand <file> <rand|seq> <qd> <nreq> [chunk_mb]\n");
        return 2;
    }
    path = argv[1];
    seq = !strcmp(argv[2], "seq");
    qd = atoi(argv[3]);
    nreq = atoi(argv[4]);
    if (argc > 5) chunk = atol(argv[5]) * 1024L * 1024L;

    struct stat st;
    if (stat(path, &st)) { perror("stat"); return 1; }
    long slots = st.st_size / chunk;
    if (nreq > slots) nreq = slots;

    offs = malloc(sizeof(long) * nreq);
    if (seq) {
        for (int i = 0; i < nreq; i++) offs[i] = (long)i * chunk;
    } else {
        // Sample WITHOUT replacement so both arms move identical bytes exactly once —
        // with replacement, a repeat could be served by the drive's own DRAM and the
        // random arm would flatter itself.
        long *perm = malloc(sizeof(long) * slots);
        for (long i = 0; i < slots; i++) perm[i] = i;
        for (long i = slots - 1; i > 0; i--) {
            long j = random() % (i + 1), t = perm[i];
            perm[i] = perm[j]; perm[j] = t;
        }
        for (int i = 0; i < nreq; i++) offs[i] = perm[i] * chunk;
        free(perm);
    }

    pthread_t th[512];
    struct timespec a, b;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (long t = 0; t < qd; t++) pthread_create(&th[t], NULL, worker, (void *)t);
    for (int t = 0; t < qd; t++) pthread_join(th[t], NULL);
    clock_gettime(CLOCK_MONOTONIC, &b);

    double s = (b.tv_sec - a.tv_sec) + (b.tv_nsec - a.tv_nsec) / 1e9;
    double gb = (double)nreq * chunk / 1e9;
    printf("%-4s qd=%-3d chunk=%6.1fMB n=%-4d %7.2f GB in %6.3f s = %6.2f GB/s\n",
           seq ? "seq" : "rand", qd, chunk / 1048576.0, nreq, gb, s, gb / s);
    return 0;
}
