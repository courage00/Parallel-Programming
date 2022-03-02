#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
#include <string>
#include <iostream>
#include <sstream>

using std::cout;
using std::endl;
using std::string;
using std::stringstream;
using namespace std;

void construct_matrices(int *n_ptr, int *m_ptr, int *l_ptr, int **a_mat_ptr, int **b_mat_ptr)
{
    // init MPI
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0)
    {
        scanf("%d%d%d",n_ptr,m_ptr,l_ptr);
        // cout << *n_ptr << endl;
        // cout << *m_ptr << endl;
        // cout << *l_ptr << endl;
        *a_mat_ptr = (int *)malloc(*n_ptr * *m_ptr *sizeof(int));  
        *b_mat_ptr = (int *)malloc(*l_ptr * *m_ptr * sizeof(int));
        //int offset =0;
        //*b_mat_ptr = (int *)malloc(*l_ptr * *m_ptr *sizeof(int));
        //給a陣列值
        for (int i = 0; i < *n_ptr; i++)
        {
            for (int j = 0; j < *m_ptr; j++)
            {
                scanf("%d", (*a_mat_ptr) + i * *m_ptr + j);
            }
        }
        for (int i = 0; i < *m_ptr; i++)
        {
            for (int j = 0; j < *l_ptr; j++)
            {
                scanf("%d", (*b_mat_ptr) + j * *m_ptr + i);
            }
        }
        // for (int i = 1; i < world_size; i++)
        // {
        //     MPI_Send(&(**b_mat_ptr), *l_ptr * *m_ptr, MPI_INT, i, 0, MPI_COMM_WORLD);
        // }
    } 
    // MPI_Bcast(n_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(m_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(l_ptr, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // if(world_rank>0)
    // {
    //     *b_mat_ptr = (int *)malloc(*l_ptr * *m_ptr * sizeof(int));
    //     MPI_Recv(&(**b_mat_ptr), *l_ptr * *m_ptr, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // }
    //MPI_Barrier(MPI_COMM_WORLD);
}
void matrix_multiply(const int n, const int m, const int l, const int *a_mat, const int *b_mat)
{
    // init MPI
    //printf("%d\n", n);
    // int world_rank, world_size;
    // MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // int local_size = n / world_size;
    // int *answer = (int *)malloc(n * l * sizeof(int));
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0)
    {
        // if (local_size != 0)
        // {
        //     int numRank = 1;
        //     for (int i = local_size; i < n; i = i + local_size)
        //     {
        //         int *local_a_mat_ptr = (int *)malloc(m * local_size * sizeof(int));
        //         if (i + local_size > n)
        //         {
        //             break;
        //         }
        //         for (int k = 0; k < local_size; k++)
        //         {
        //             for (int j = 0; j < m; j++)
        //             {
        //                 local_a_mat_ptr[k * m + j] = a_mat[i * k * m + j];
        //             }
        //         }
        //         MPI_Send(&(*local_a_mat_ptr), local_size * m, MPI_INT, numRank, 0, MPI_COMM_WORLD);
        //     }
        // }
        // else//超小矩陣答案
        // {
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < l; k++)
                {
                    int temp = 0;
                    for (int j = 0; j < m; j++)
                    {
                        temp += a_mat[i * m + j] * b_mat[k * m + j];
                    }
                    if (k != (l - 1))
                    {
                        printf("%d", temp);
                        printf(" ");
                    }
                    else
                    {
                        printf("%d", temp);
                        printf("\n");
                    }
                }
            }
        //}
    }
    // else
    // {
    //     if (local_size != 0)
    //     {
    //         int *local_a_mat_ptr = (int *)malloc(m * local_size * sizeof(int));
    //         MPI_Recv(&(*local_a_mat_ptr), local_size * m, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //         // for (int k = 0; k < local_size; k++)
    //         // {
    //         //     for (int j = 0; j < m; j++)
    //         //     {
    //         //         printf("%d _", local_a_mat_ptr[k * m + j]);
    //         //     }
    //         //      printf("\n");
    //         // }
    //         for (int i = 0; i < n; i++)
    //         {
    //             for (int k = 0; k < l; k++)
    //             {
    //                 int temp = 0;
    //                 for (int j = 0; j < m; j++)
    //                 {
    //                     temp += a_mat[i * m + j] * b_mat[k * m + j];
    //                 }
    //             }
    //         }
    //     }
    // }
    //     for (int i = 0; i < l; i++)
    //     {
    //         for (int j = 0; j < m; j++)
    //         {
    //             printf("%d ", b_mat[ i* m + j]);
    //         }
    //         printf("\n");
    //     }

    // // if (world_rank == 0)
    // // {
    // //     for (int i = 0; i < n; i++)
    // //     {
    // //         for (int j = 0; j < l; j++)
    // //         {
    // //             printf("%d ", C[ i * l + j]);
    // //         }
    // //         printf("\n");
    // //     }
    // // }
    // // MPI_Scatter(Matrix_one, local_M * N, MPI_INT, local_Matrix_one, local_M * N, MPI_INT, 0, MPI_COMM_WORLD);
    // // MPI_Bcast(Matrix_two, M * N, MPI_INT, 0, MPI_COMM_WORLD);
    // // for (i = 0; i < local_M; i++)
    // //     for (j = 0; j < M; j++)
    // //     {
    // //         tem = 0;
    // //         for (k = 0; k < N; k++)
    // //             tem += local_Matrix_one[i * M + k] * Matrix_two[j * M + k];
    // //         local_result[i * M + j] = tem;
    // //     }
    // // free(local_Matrix_one);
    // // result_Matrix = (int *)malloc(M * N * sizeof(int));
    // // MPI_Gather(local_result, local_M * N, MPI_INT, result_Matrix, local_M * N, MPI_INT, 0, MPI_COMM_WORLD);
}
void destruct_matrices(int *a_mat, int *b_mat)
{
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0)
    {
        free(a_mat);
        free(b_mat);
    }
}
