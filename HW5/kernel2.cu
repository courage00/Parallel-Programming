#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm> 

__global__ void mandelKernel(float gLowerX, float gLowerY,float gStepX, float gStepY, int gMaxIterations,int* gResult) 
{
    int a =blockIdx.x *blockDim.x + threadIdx.x;
    int i = a%1600;
    int j = a/1600;
    float x = gLowerX + i * gStepX;
    float y = gLowerY + j * gStepY;
    float z_re = x; 
    float z_im = y;
    float  temp;
    int k=0;
    while((z_re * z_re + z_im * z_im <= 4)&& (k < gMaxIterations))
    {

        temp = z_re * z_re - z_im * z_im+x;
        //new_im = 2.f * z_re * z_im;
        z_im = y + 2.f * z_re * z_im;
        z_re = temp;
        k++;
    }
         gResult[a]=k;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;

    //配置主機記憶體
    int *pResult;
    // pResult = (int *)malloc(resX*resY*sizeof(int));
    cudaHostAlloc( (void**)&pResult,resX*resY*sizeof(int),cudaHostAllocDefault);
    //配置顯示卡記憶體
    int  *gResult;
    size_t pitch;
    cudaMallocPitch((void**)&gResult, &pitch,  1875 * sizeof(int), 1024);

    // //嘗試 zero copy
    // cudaHostGetDevicePointer(&gResult, pResult, 0);
    //cudaMallocPitch((void**)&gResult, &pitch,  resX*resY * sizeof(int), 1);
    //printf("%f %f %f %f %d\n",*pLowerX,*pLowerY,*pStepX,*pStepY,*pMaxIterations);
    //printf("%lu\n",pitch);
    dim3 threadsPerBlock(32);
    dim3 numBlocks(60000);
    mandelKernel<<<numBlocks,threadsPerBlock>>>(lowerX,lowerY,stepX,stepY,maxIterations,gResult);
    cudaMemcpy(pResult, gResult, resX*resY*sizeof(int), cudaMemcpyDeviceToHost);

    //給回img
    std::copy(pResult, pResult+1920000,img);

    // //釋放記憶體
    cudaFreeHost(pResult);
    cudaFree(gResult);

}
