__device__ int get(int x, int y,int width){
  return y * width +x;
}

__device__ int normeValue(int x, int width){
	if(x < 0) //-1
		return width - 1;
	if(x == width)
		return 0;
	return x;
}

__device__ void neighborsIndexes(int i, int j, int width, int height, int* dir) {
    dir[0] = get(normeValue(i + 1, width), j, width);
    dir[1] = get(normeValue(i + 1, width), normeValue(j + 1, height), width);
    dir[2] = get(i, normeValue(j + 1, height), width);
    dir[3] = get(normeValue(i - 1, width), normeValue(j + 1, height), width);
    dir[4] = get(normeValue(i - 1, width), j, width);
    dir[5] = get(normeValue(i - 1, width), normeValue(j - 1, height), width);
    dir[6] = get(i, normeValue(j - 1, height), width);
    dir[7] = get(normeValue(i + 1, width), normeValue(j - 1, height), width);
}


__device__ float getTotalUpdateFromNeighbors(float* tmp, int i, int j, int width, int height){
	int iPlusOne = i + 1;
	int jPlusOne = j + 1;
 	int iMinusOne = i - 1;
 	int jMinusOne = j - 1;
       return 
        tmp[get(normeValue(iPlusOne,width), j, width)] +
        tmp[get(normeValue(iPlusOne,width), normeValue(jPlusOne,height),width)] +
	tmp[get(i, normeValue(jPlusOne,height),width)] +
        tmp[get(normeValue(iMinusOne,width), normeValue(jPlusOne,height),width)] +
        tmp[get(normeValue(iMinusOne,width), j, width)] +
        tmp[get(normeValue(iMinusOne,width), normeValue(jMinusOne,height),width)] +
        tmp[get(i, normeValue(jMinusOne,height),width)] +
        tmp[get(normeValue(iPlusOne,width), normeValue(jMinusOne,height),width)];
}


__device__ int getNumberOfNoBlockNeighbors(float* values, int i, int j, int width, int height){
	int iPlusOne = i + 1;
	int jPlusOne = j + 1;
 	int iMinusOne = i - 1;
 	int jMinusOne = j - 1;
 	int nb = 8;
 	if(values[get(normeValue(iPlusOne,width), j, width)] < 0) nb--;
 	if(values[get(normeValue(iPlusOne,width), normeValue(jPlusOne,height),width)] < 0) nb --;
 	if(values[get(i, normeValue(jPlusOne,height),width)]  < 0) nb --;
 	if(values[get(normeValue(iMinusOne,width), normeValue(jPlusOne,height),width)]  < 0) nb --;
 	if(values[get(normeValue(iMinusOne,width), j, width)]  < 0) nb --;
 	if(values[get(normeValue(iMinusOne,width), normeValue(jMinusOne,height),width)]  < 0) nb --;
 	if(values[get(i, normeValue(jMinusOne,height),width)]  < 0) nb --;
 	if(values[get(normeValue(iPlusOne,width), normeValue(jMinusOne,height),width)] < 0) nb --;
 	return nb;
}

extern "C"
__global__ void DIFFUSION_TO_TMP( int width, int height, float *values, float* tmp, float diffCoef)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
       
       //filling tmp
       if (i < width && j < height ){
    		int k = get(i,j,width);
    		if(values[k] == -1){
    			tmp[k] = 0;
    			return;
			}
	        //int nb = getNumberOfNoBlockNeighbors(values, i, j, width, height);
    		float give = values[k] * diffCoef;
    		float giveToNeighbor = give / 8;
    		values[k] -= give;//TODO a[k] = value - give
    		tmp[k] = giveToNeighbor;
        }
}


extern "C"
__global__ void DIFFUSION_UPDATE( int width, int height, float *values, float* tmp)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){//TODO + alone
       int index = get(i,j,width);
       if(values[index] != -1)
    		values[index] += getTotalUpdateFromNeighbors(tmp, i, j, width, height);
    	}
}


extern "C"
__global__ void DIFFUSION_UPDATE_THEN_EVAPORATION( int width, int height, float *values, float* tmp, float evapCoef)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){//TODO + alone
        int k = get(i,j,width);
        if(values[k] == -1)
        	return;
        float total = values[k] + getTotalUpdateFromNeighbors(tmp, i, j, width, height);
    	values[k] = total - total * evapCoef;
    	}
}



