#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

pthread_mutex_t mutex;
long long int sum = 0;

typedef struct
{
   int threadID;
   long long int partTossingNUM;
} threadInfo; // 紀錄thread的資訊

void *pTossing(void *param){

	long long int numInCircle = 0;
	float x=0.f,y=0.f;
    threadInfo *t = (threadInfo *)param;
    int tossingNUM = t->partTossingNUM;
	unsigned int seed = time(NULL)+t->threadID*6000;//隨機性
    //unsigned int seed2 = seed+t->threadID;
    //  printf("pnum : %d\n", tossingNUM);
    //unsigned int rNUM = rand_r(&seed);
	for(int i = 0;i < tossingNUM;i++){
		x = (float) rand_r(&seed) / RAND_MAX;
		y = (float) rand_r(&seed) / RAND_MAX;

		if (x * x + y * y <= 1){
		       numInCircle++;
		}
	}

	pthread_mutex_lock(&mutex);//互斥存取
	sum += numInCircle;
    // printf("pID : %d\n", tID);//確認thread數正確
	pthread_mutex_unlock(&mutex);

	pthread_exit(NULL);	
}

int main(int argc, char *argv[])
{
    pthread_mutex_init(&mutex, NULL);
    long long int CPUNum = atoi(argv[1]);
    long long int numTossing = atoi(argv[2]);
    long long int partTossing = numTossing / CPUNum;

    pthread_t *th;
    th = (pthread_t *)malloc(CPUNum * sizeof(pthread_t));
    // pthread_attr_t attr;
    // pthread_attr_init(&attr);
    // pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);


    pthread_mutex_init(&mutex, NULL);//互斥初始化
    threadInfo tInfo[CPUNum];

    // time_t startTime, endTime;
    // startTime = time (NULL);
    for(int i = 0;i < CPUNum;i++){
	    tInfo[i].threadID = i;
        tInfo[i].partTossingNUM = partTossing;
	    pthread_create(&th[i], NULL, pTossing, (void *)&tInfo[i]);
    }
    for(int i = 0;i < CPUNum; i++){
	    pthread_join(th[i], NULL);
    }
    // endTime = time (NULL);
    // double diffTime = difftime(endTime, startTime);
    printf("%f\n", (4.0 * (double)sum / (double)numTossing));
    // printf("time : %f\n", diffTime);
    pthread_mutex_destroy(&mutex);
    //free(th);
    return 0;
}