#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

__global__ void mandelKernel(float gLowerX, float gLowerY,float gStepX, float gStepY, int gMaxIterations,int* gResult,int no) {   
    int a =(no*30000+blockIdx.x) *blockDim.x + threadIdx.x;
    //int a =blockIdx.x *blockDim.x + threadIdx.x;
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
    // cudaMemcpyToSymbol(c_lowerX, &lowerX, sizeof(float));
    // cudaMemcpyToSymbol(c_lowerY, &lowerY, sizeof(float));
    // cudaMemcpyToSymbol(c_stepX, &stepX, sizeof(float));
    // cudaMemcpyToSymbol(c_stepY, &stepY, sizeof(float));
    // cudaMemcpyToSymbol(c_maxIterations, &maxIterations, sizeof(int));
    //配置主機記憶體
    int *pResult;
    //pResult = (int *)malloc(resX*resY*sizeof(int));
    cudaHostAlloc( (void**)&pResult,resX*resY*sizeof(int),cudaHostAllocDefault);
    //cudaMallocManaged(&pResult, resX*resY* sizeof(int));

    //配置顯示卡記憶體
    // float *gLowerX, *gLowerY, *gStepX, *gStepY;
    // int *gMaxIterations, *gResult;
    int  *gResult;
    // cudaHostAlloc((void **)&gLowerX, sizeof(float));
    // cudaHostAlloc((void **)&gLowerY, sizeof(float));
    // cudaHostAlloc((void **)&gStepX, sizeof(float));
    // cudaHostAlloc((void **)&gStepY, sizeof(float));
    // cudaHostAlloc((void **)&gMaxIterations, sizeof(int));

    // cudaMalloc((void **)&lowerX, sizeof(float));
    // cudaMalloc((void **)&lowerY, sizeof(float));
    // cudaMalloc((void **)&stepX, sizeof(float));
    // cudaMalloc((void **)&stepY, sizeof(float));
    // cudaMalloc((void **)&maxIterations, sizeof(int));
    cudaMalloc((void **)&gResult, resX * resY * sizeof(int));

    //cudaHostRegister (img,resX*resY*sizeof(int),cudaHostRegisterMapped);

     //size_t pitch;
    //cudaMallocPitch((void**)&gResult, &pitch,  15000 * sizeof(int), 128);
    //cudaMallocPitch((void**)&gResult, &pitch,  30000 * sizeof(int), 256);
    //cudaMallocPitch((void**)&gResult, &pitch,  1875 * sizeof(int), 1024);
    //printf("%f %f %f %f %d\n",*pLowerX,*pLowerY,*pStepX,*pStepY,*pMaxIterations);
    //printf("%lu\n",pitch);
    //載入到顯示卡記憶體中
    // cudaMemcpy(gLowerX, &lowerX, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(gLowerY, &lowerY, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(gStepX, &stepX, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(gStepY, &stepY, sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(gMaxIterations, &maxIterations, sizeof(int), cudaMemcpyHostToDevice);
    //cudaMemcpy(gResult, pResult, resX*resY*sizeof(int), cudaMemcpyHostToDevice);

    //嘗試 zero copy
    // cudaHostAlloc((void **)&pResult,resX * resY * sizeof(int), cudaHostAllocMapped);
    // //cudaHostGetDevicePointer(&gResult, img, 0);
    // cudaHostGetDevicePointer(&gResult, pResult, 0);


    // dim3 threadsPerBlock(1024);
    // dim3 numBlocks(1875);
    // dim3 threadsPerBlock(512);
    // dim3 numBlocks(3750);
    dim3 threadsPerBlock(32);//32 256 
    dim3 numBlocks(30000);//60000 7500
    //mandelKernel<<<numBlocks,threadsPerBlock>>>(gLowerX,gLowerY,gStepX,gStepY,gMaxIterations,gResult);
    //mandelKernel<<<numBlocks,threadsPerBlock>>>(lowerX,lowerY,stepX,stepY,maxIterations,gResult);
    //mandelKernel<<<numBlocks,threadsPerBlock>>>(gResult);
    //cudaMemcpy(img, gResult, resX*resY*sizeof(int), cudaMemcpyDeviceToHost);
       
    //asyn 複製
    int  nStreams =2;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i)
    cudaStreamCreate(&stream[i]);
    int nsize = 1920000/nStreams;
    for (int i = 0; i < nStreams; ++i)
    {
        //int offset = i * streamSize;
        //kernel<<<streamSize / blockSize, blockSize, 0, stream[i]>>>(d_a, offset);
        mandelKernel<<<numBlocks, threadsPerBlock, 0, stream[i]>>>(lowerX, lowerY, stepX, stepY, maxIterations, gResult, i);
        // int offset = i * nsize;
        // cudaMemcpyAsync(&img[offset], &gResult[offset],
        //                 nsize * sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
    }
    for (int i = 0; i < nStreams; ++i)
    {
        int offset = i * nsize;
        cudaMemcpyAsync(&img[offset], &gResult[offset],
                        nsize*sizeof(int), cudaMemcpyDeviceToHost, stream[i]);
    }

    //同步
    //cudaThreadSynchronize();
    //cudaDeviceSynchronize();
    //cpuFunction();
    //給回img
    //std::copy(pResult, pResult+1920000,img);//方一
    // for(int k=0;k<1920000;k++) //方二
    //     img[k]=pResult[k];
    //memcpy(img, pResult, 1920000*sizeof(int));//方三
    //printf(" %d  %d  %d  %d %d\n",pResult[0],pResult[1],pResult[55550],img[0],img[55550]);

    // //釋放記憶體
    //free(pResult);

    cudaFree(gResult);
    cudaFreeHost(pResult);
}

