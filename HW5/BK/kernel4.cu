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
    //pResult = (int *)malloc(resX*resY*sizeof(int));
    cudaMallocManaged(&pResult, resX*resY* sizeof(int));

    dim3 threadsPerBlock(32);
    dim3 numBlocks(60000);//60000
    //mandelKernel<<<numBlocks,threadsPerBlock>>>(gLowerX,gLowerY,gStepX,gStepY,gMaxIterations,gResult);
    mandelKernel<<<numBlocks,threadsPerBlock>>>(lowerX,lowerY,stepX,stepY,maxIterations,pResult);
    //mandelKernel<<<numBlocks,threadsPerBlock>>>(gResult);
    //cudaMemcpy(img, gResult, resX*resY*sizeof(int), cudaMemcpyDeviceToHost);

    //asyn 複製
    // int  nStreams =2;
    // cudaStream_t stream[nStreams];
    // for (int i = 0; i < nStreams; ++i)
    // cudaStreamCreate(&stream[i]);
    // for (int i = 0; i < nStreams; ++i)
    // {
    //     mandelKernel<<<numBlocks, threadsPerBlock, 0, stream[i]>>>(lowerX, lowerY, stepX, stepY, maxIterations, pResult, i);
    // }

    //同步
    cudaDeviceSynchronize();
    //給回img
    std::copy(pResult, pResult+1920000,img);//方一
    // for(int k=0;k<1920000;k++) //方二
    //     img[k]=pResult[k];
    //memcpy(img, pResult, 1920000*sizeof(int));//方三
    //printf(" %d  %d  %d  %d %d\n",pResult[0],pResult[1],pResult[55550],img[0],img[55550]);

    // //釋放記憶體
    //free(pResult);

    //cudaFree(gResult);
    cudaFreeHost(pResult);

}

