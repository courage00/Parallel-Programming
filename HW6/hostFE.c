#include <stdio.h>
#include <stdlib.h>
#include "hostFE.h"
#include "helper.h"



cl_command_queue commands; // compute command queue
cl_kernel kernel;          // compute kernel
cl_mem input;              // device memory used for the input array
cl_mem output;             // device memory used for the output array
cl_mem cFilter;
cl_mem cFilterWidth; 
size_t global;                      // global domain size for our calculation

void hostFE(int filterWidth, float *filter, int imageHeight, int imageWidth,
            float *inputImage, float *outputImage, cl_device_id *device,
            cl_context *context, cl_program *program)
{
    cl_int status;
    int filterSize = filterWidth * filterWidth* sizeof(float);
    int imgSize =imageHeight*imageWidth* sizeof(float);


    commands = clCreateCommandQueue(*context, *device, 0, 0);
    kernel = clCreateKernel(*program, "convolution", 0);
    // Create the input and output arrays in device memory for our calculation
    input = clCreateBuffer(*context,  CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,  imgSize, inputImage, NULL);
    output = clCreateBuffer(*context, CL_MEM_WRITE_ONLY| CL_MEM_COPY_HOST_PTR, imgSize, outputImage, NULL);
    cFilter = clCreateBuffer(*context,  CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,  filterSize, filter, NULL);;
    cFilterWidth = clCreateBuffer(*context,  CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,  sizeof(int), &filterWidth, NULL);; 
                //    printf("ok2\n");
    // Set the arguments to our compute kernel
    //
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &cFilter);
    //clSetKernelArg(kernel, 3, sizeof(int), &filterWidth);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &cFilterWidth);

    // clGetKernelWorkGroupInfo(kernel, *device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    // printf("%d\n", local);//256
    global = imageHeight*imageWidth;
    //測試2d szie
    // size_t globalWorkSize[2];
    // globalWorkSize[0]= 600;
    // globalWorkSize[1]=400;
    // size_t localWorkSize[2];
    // localWorkSize[0]=4;
    // localWorkSize[1]=4;


    //clEnqueueNDRangeKernel(commands, kernel, 2, 0, globalWorkSize, localWorkSize, 0, 0, 0); // NULL  or &local // better NULL
    clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, NULL, 0, NULL, NULL); // NULL  or &local // better NULL
    
    clFinish(commands);
    clEnqueueReadBuffer( commands, output, CL_TRUE, 0, imgSize, outputImage, 0, NULL, NULL );

    clReleaseKernel(kernel);
    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseMemObject(cFilter);
    clReleaseCommandQueue(commands);
}