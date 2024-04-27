#include <stdio.h>
#include <stdlib.h>

int randomRange(int min, int max) {
    int value = rand() % (max + 1 - min) + min;
    return value;
}

int main(int argc, char *argv[]) {
    int min = 1;
    int max = 10;

    int sleepTime = randomRange(min, max);

    fprintf(stderr, "Start a new Process.......\n");

}