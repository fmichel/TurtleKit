__device__ int get(int x, int y,int width){
  return y * width +x;
}

extern "C"
__global__ void EVAPORATION( int width, int height, float *values, float evapCoef)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){//TODO + alone
        int k = get(i,j,width);
    	values[k] -= values[k] * evapCoef;
    	}
}

