__kernel void convolution(const __global float *inputImage, __global float *outputImage,
 const __global float *filter,__global int *filterWidth)
{
    int halffilterSize = *filterWidth / 2;
    float  sum = 0;
    int k, l;
    int shPlus = (*filterWidth-3)/2+1;//偏移量+1
    int index =get_global_id(0);
    int j = index%600;
    int i = index/600;

        for (k = -1; k <= 1; k++)
        {
            for (l = -1; l <= 1; l++)
            {
                    sum += inputImage[(i + k) * 600 + j + l] *
                           filter[(k + shPlus) * *filterWidth +
                                  l + shPlus];
            }
        }
    outputImage[i * 600 + j] = sum;
}


