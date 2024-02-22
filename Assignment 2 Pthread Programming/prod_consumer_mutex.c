/*
 * This problem has you solve the classic "bounded buffer" problem with
 * one producer and multiple consumers:
 *
 *                                       ------------
 *                                    /->| consumer |
 *                                    |  ------------
 *                                    |
 *  ------------                      |  ------------
 *  | producer |-->   bounded buffer --->| consumer |
 *  ------------                      |  ------------
 *                                    |
 *                                    |  ------------
 *                                    \->| consumer |
 *                                       ------------
 *
 *  The program below includes everything but the implementation of the
 *  bounded buffer itself.  main() does this:
 *
 *  1. starts N consumers as per the first argument (default 1)
 *  2. starts the producer
 *
 *  The producer reads positive integers from standard input and passes those
 *  into the buffer.  The consumers read those integers and "perform a
 *  command" based on them (all they really do is sleep for some period...)
 *
 *  on EOF of stdin, the producer passes N copies of -1 into the buffer.
 *  The consumers interpret -1 as a signal to exit.
 */

#include <stdio.h>
#include <stdlib.h>             /* atoi() */
#include <unistd.h>             /* usleep() */
#include <assert.h>             /* assert() */
#include <signal.h>             /* signal() */
#include <alloca.h>             /* alloca() */
#include <pthread.h>

void print_insertion(int number, int location) {
    printf("producer inserting %d at location %d\n", number, location);
}

void print_extraction(int consumerno, int number, int location) {
	printf("consumer %d extract %d from %d\n", consumerno, number, location);
}

/**************************************************************************\
 *                                                                        *
 * Bounded buffer.  This is the only part you need to modify.  Your       *
 * buffer should have space for up to 10 integers in it at a time.        *
 *                                                                        *
 * Add any data structures you need (globals are fine) and fill in        *
 * implementations for these three procedures:                            *
 *                                                                        *
 * void buffer_init(void)                                                 *
 *                                                                        *
 *      buffer_init() is called by main() at the beginning of time to     *
 *      perform any required initialization.  I.e. initialize the buffer, *
 *      any mutex/condition variables, etc.                               *
 *                                                                        *
 * void buffer_clean(void)                                                *
 *                                                                        *
 *      buffer_clean() is called by main() at the beginning of time to    *
 *      perform any required cleanup.  I.e. cleanup the buffer,           *
 *      any mutex/condition variables, etc.                               *
 *                                                                        *
 * void buffer_insert(int number)                                         *
 *                                                                        *
 *      buffer_insert() inserts a number into the next available slot in  *
 *      the buffer.  If no slots are available, the thread should wait    *
 *      (not spin-wait!) for an empty slot to become available.           *
 *                                                                        *
 * int buffer_extract(int consumerno)                                     *
 *                                                                        *
 *      buffer_extract() removes and returns the number in the next       *
 *      available slot.  If no number is available, the thread should     *
 *      wait (not spin-wait!) for a number to become available.  Note     *
 *      that multiple consumers may call buffer_extract() simulaneously.  *
 *                                                                        *
\**************************************************************************/

/* DO NOT change MAX_BUF_SIZE */
#define MAX_BUF_SIZE    10

int buffer[MAX_BUF_SIZE] = {0};
int location = 0;

pthread_mutex_t mutexKey;

void buffer_init(void) {
    //printf("buffer_init called: doing nothing\n"); /* FIX ME */
    pthread_mutex_init(&mutexKey, NULL);
}

void update_buffer(char operation, int number, int *locationNumberArr){
    // Lock a mutex before updating buffer[] and location
    pthread_mutex_lock(&mutexKey);

    if (operation == 'i') {
        if(location < MAX_BUF_SIZE){
            if(number != -1 || location == 0){
                locationNumberArr[0] = location;
                locationNumberArr[1] = number;

                // Insert number to buffer[] at specific location
                buffer[location] = number;
                // Increase location
                location++;
            }else{
                buffer[location] = number;

                // Insert -1 to buffer[0]
                int index = -10;
                for(int i=0; i<=location; i++){
                    int temp = buffer[i];
                    if(temp > 0){
                        index = i;
                        break;
                    }
                }

                if(index == -10){
                    locationNumberArr[0] = location;
                    locationNumberArr[1] = number;
                }else{
                    locationNumberArr[0] = 0;
                    locationNumberArr[1] = number;

                    int temp = buffer[index];
                    buffer[index] = buffer[location];
                    buffer[location] = temp;
                }

                // Increase location
                location++;
            }
        }else{
            locationNumberArr[0] = -10;
            locationNumberArr[1] = number;
        }

    } else {
        if(location > 0){
            // Decrease location
            location--;

            // Extract number from buffer[] at specific location
            number = buffer[location];

            locationNumberArr[0] = location;
            locationNumberArr[1] = number;
        }else{
            locationNumberArr[0] = -10;
            locationNumberArr[1] = number;
        }
    }

    // Unlock it after updating buffer[] and location
    pthread_mutex_unlock(&mutexKey);
}

void buffer_insert(int number) {
    //int location = -1;

    int locationNumberArr[2] = {0};

    update_buffer('i', number, locationNumberArr);
    while(locationNumberArr[0] < 0){
        usleep(10000);

        update_buffer('i', number, locationNumberArr);
    }

    print_insertion(number, locationNumberArr[0]);
}

int buffer_extract(int consumerno) {
    int number = -1;
    //int location = -1;

    int locationNumberArr[2] = {0};

    update_buffer('e', -10, locationNumberArr);
    while(locationNumberArr[0] < 0){
        usleep(10000);

        update_buffer('e', -10, locationNumberArr);
    }

    number = locationNumberArr[1];

    print_extraction(consumerno, number, locationNumberArr[0]);

    return number;                   /* FIX ME */
}

void buffer_clean(void) {
    //printf("buffer_clean called: doing nothing\n"); /* FIX ME */
    pthread_mutex_destroy(&mutexKey);
    pthread_exit(NULL);
}

/**************************************************************************\
 *                                                                        *
 * consumer thread.  main starts any number of consumer threads.  Each    *
 * consumer thread reads and "interprets" numbers from the bounded        *
 * buffer.                                                                *
 *                                                                        *
 * The interpretation is as follows:                                      *
 *                                                                        *
 * o  positive integer N: sleep for N * 100ms                             *
 * o  negative integer:  exit                                             *
 *                                                                        *
 * Note that a thread procedure (called by pthread_create) is required to *
 * take a void * as an argument and return a void * as a result.  We play *
 * a dirty trick: we pass the thread number (main()'s idea of the thread  *
 * number) as the "void *" argument.  Hence the casts.  The return value  *
 * is ignored so we return NULL.                                          *
 *                                                                        *
\**************************************************************************/

void *consumer_thread(void *raw_consumerno) {
    int consumerno = (int)(intptr_t)raw_consumerno; /* dirty trick to pass in an integer */

    printf("  consumer %d: starting\n", consumerno);
    while (1) {
        int number = buffer_extract(consumerno);

        if (number < 0)
            break;

        usleep(10000 * number);  /* "interpret" the command */
        fflush(stdout);
    }

    printf("  consumer %d: exiting\n", consumerno);
    return(NULL);
}

/**************************************************************************\
 *                                                                        *
 * producer.  main calls the producer as an ordinary procedure rather     *
 * than creating a new thread.  In other words the original "main" thread *
 * becomes the "producer" thread.                                         *
 *                                                                        *
 * The producer reads numbers from stdin, and inserts them into the       *
 * bounded buffer.  On EOF from stdin, it finished up by inserting a -1   *
 * for every consumer thread so that all the consumer thread shut down    *
 * cleanly.                                                               *
 *                                                                        *
\**************************************************************************/

#define MAXLINELEN 128

void producer(int nconsumers) {
    char buffer[MAXLINELEN];
    int number;

    printf("  producer: starting\n");

    while (fgets(buffer, MAXLINELEN, stdin) != NULL) {
        number = atoi(buffer);
        buffer_insert(number);
    }

    printf("producer: read EOF, sending %d '-1' numbers\n", nconsumers);
    for (number = 0; number < nconsumers; number++)
        buffer_insert(-1);

    printf("producer: exiting\n");
}

/*************************************************************************\
 *                                                                       *
 * main program.  Main calls buffer_init and does other initialization,  *
 * fires of N copies of the consumer (N given by command-line argument), *
 * then calls the producer.                                              *
 *                                                                       *
\*************************************************************************/

int main(int argc, char *argv[]) {
    pthread_t *consumers;
    int nconsumers = 1;
    int kount;

    if (argc > 1)
        nconsumers = atoi(argv[1]);

    printf("main: nconsumers = %d\n", nconsumers);
    assert(nconsumers >= 0);

    /*
     * 1. initialization
     */
    buffer_init();
    signal(SIGALRM, SIG_IGN);     /* evil magic for usleep() under solaris */

    /*
     * 2. start up N consumer threads
     */
    consumers = (pthread_t *)alloca(nconsumers * sizeof(pthread_t));
    for (kount = 0; kount < nconsumers; kount++) {
        int test = pthread_create(&consumers[kount], /* pthread number */
                NULL,            /* "attributes" (unused) */
                consumer_thread, /* procedure */
                (void *)(intptr_t)kount);  /* hack: consumer number */

        assert(test == 0);
    }

    /*
     * 3. run the producer in this thread.
     */
    producer(nconsumers);

    /*
     * n. clean up: the producer told all the consumers to shut down (by
     *    sending -1 to each).  Now wait for them all to finish.
     */
    for (kount = 0; kount < nconsumers; kount++) {
        int test = pthread_join(consumers[kount], NULL);

        assert(test == 0);
    }
    return(0);
}
