__device__ int get(int x, int y,int width){
  return y * width +x;
}

__device__ float getTotalUpdateFromNeighbors(float* tmp, int* neighborsIndexes){
	float sum = 0;
	for(int i = 0; i < 8; i++){
		sum += tmp[neighborsIndexes[i]]; 
	}
   return sum;
}

extern "C"
__global__ void DIFFUSION_TO_TMP( int width, int height, float *values, float* tmp, float diffCoef)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
       
       //filling tmp
       if (i < width && j < height ){
    		int k = get(i,j,width);
    		float give = values[k] * diffCoef;
    		float giveToNeighbor = give / 8;
    		values[k] -= give;//TODO a[k] = value - give
    		tmp[k] = giveToNeighbor;
        }
}


extern "C"
__global__ void DIFFUSION_UPDATE( int width, int height, float *values, float* tmp, int* neighborsIndexes)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){
	    	values[get(i,j,width)] += getTotalUpdateFromNeighbors(tmp, neighborsIndexes + get(i,j,width)*8);
    	}
}


extern "C"
__global__ void DIFFUSION_UPDATE_THEN_EVAPORATION( int width, int height, float *values, float* tmp, float evapCoef, int* neighborsIndexes)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){//TODO + alone
        int k = get(i,j,width);
        float total = values[k] + getTotalUpdateFromNeighbors(tmp, neighborsIndexes + get(i,j,width)*8);
    	values[k] = total - total * evapCoef;
    	}
}

