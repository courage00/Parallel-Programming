#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv)
{
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank( MPI_COMM_WORLD , &world_rank);
    MPI_Comm_size( MPI_COMM_WORLD , &world_size );
    long long int partTossing = tosses/world_size;
    long long int numInCircle = 0;
    long long int sum = 0;
    float x = 0.f, y = 0.f;
    double distanceSquared=0;
    unsigned int seed = time(NULL) + world_rank * world_rank * 6000; //隨機性

    for (int i = 0; i < partTossing; i++)
    {
        x = (float)rand_r(&seed) / RAND_MAX;
        y = (float)rand_r(&seed) / RAND_MAX;
        distanceSquared=x * x + y * y;
        if (distanceSquared <= 1)
        {
            numInCircle++;
        }
    }

    if (world_rank == 0)
    {
        // Master
        long long int *remoteNum;
        //MPI_Win_create(&remoteSum,sizeof(long long int), sizeof(long long int), MPI_INFO_NULL,MPI_COMM_WORLD, &win);
        MPI_Alloc_mem(world_size * sizeof(long long int), MPI_INFO_NULL, &remoteNum);
        for (int i = 0; i < world_size; i++)
        {
            remoteNum[i] = 0;
        }
        MPI_Win_create(remoteNum,  world_size* sizeof(long long int), sizeof(long long int), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
        sum+=numInCircle;
       int ready = 0;
       //int i =1;
       while (!ready)
       {
          // Without the lock/unlock schedule stays forever filled with 0s
          MPI_Win_lock(MPI_LOCK_SHARED, 0, 0, win);
          for (int i= 1; i < world_size; i++)
          {
              if(remoteNum[i]==0)
              {
                  ready=0;
                  break;
              }
              else
              {
                  ready=1;
              }
              
          }
          MPI_Win_unlock(0, win);
       }
       for (int i = 1; i < world_size; i++)
       {
           sum += remoteNum[i];
       }
       MPI_Free_mem(remoteNum);
    }
    else
    {
        // Workers
       MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

       // Register with the master
       MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
       MPI_Put(&numInCircle, 1, MPI_LONG_LONG, 0, world_rank, 1, MPI_LONG_LONG, win);
       MPI_Win_unlock(0, win);
        //MPI_Accumulate(&numInCircle,1,MPI_LONG_LONG,0,world_rank,1,MPI_LONG_LONG,MPI_SUM,win);
        //MPI_Recv(&numInCircle, 1, MPI_LONG_LONG, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    
    if (world_rank == 0)
    {
        // TODO: handle PI result
        pi_result=(4.0 * (double)sum / (double)tosses);
        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }
    MPI_Win_free(&win);
    MPI_Finalize();
    return 0;
}