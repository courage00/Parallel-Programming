#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

__global__ void mandelKernel(float gLowerX, float gLowerY,float gStepX, float gStepY, int gMaxIterations,int* gResult) {   
    int a =blockIdx.x *blockDim.x + threadIdx.x;
    int i = a%1600;
    int j = a/1600;
    float x = gLowerX + i * gStepX;
    float y = gLowerY + j * gStepY;
    float z_re = x; 
    float z_im = y;
    int k;
    for (k = 0; k < (gMaxIterations); ++k)
    {
        float z = z_re * z_re + z_im * z_im;
        if (z > 4.f)
        {
            break;
        }

        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = x + new_re;
        z_im = y + new_im;
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
    pResult = (int *)malloc(resX*resY*sizeof(int));

    //配置顯示卡記憶體
    int  *gResult;
    cudaMalloc((void **)&gResult, resX * resY * sizeof(int));

    dim3 threadsPerBlock(32);//32 256 
    dim3 numBlocks(60000);//60000 7500
    mandelKernel<<<numBlocks,threadsPerBlock>>>(lowerX,lowerY,stepX,stepY,maxIterations,gResult);
    cudaMemcpy(img, gResult, resX*resY*sizeof(int), cudaMemcpyDeviceToHost);
 

    cudaFree(gResult);
    cudaFreeHost(pResult);

}

