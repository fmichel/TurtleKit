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

__device__ int* neighborsIndexes(int i, int j, int width, int height){
	int dir[8];
	dir[0] = get(normeValue(i+1,width), j, width);
	dir[1] = get(normeValue(i+1,width), normeValue(j+1,height),width);
	dir[2] = get(i, normeValue(j+1,height),width);
	dir[3] = get(normeValue(i-1,width), normeValue(j+1,height),width);
	dir[4] = get(normeValue(i-1,width), j, width);
	dir[5] = get(normeValue(i-1,width), normeValue(j-1,height),width);
	dir[6] = get(i, normeValue(j-1,height),width);
	dir[7] = get(normeValue(i+1,width), normeValue(j-1,height),width);
	return dir;
}

/*
__device__ float getTotalUpdateFromNeighbors(float* tmp, int i, int j, int width, int height){
		int index = get(i,j,width) * 8;
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
*/


extern "C"
__global__ void FIELD_MAX_DIR(int width, int height, float *values, int* patchMax)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){
    		int k = get(i,j,width);
    		int maxIndex = 0;
    		int* neighbors = neighborsIndexes(i,j,width,height);
    		float max = values[neighbors[0]];
    		for(int u=1 ; u < 8 ; u++){
    			float current = values[neighbors[u]];
    			if(max < current){
    				max = current;
    				maxIndex = u;
    			}
    		}
    		patchMax[k] = maxIndex * 45;
    	}
}
 
//with fields
extern "C"
__global__ void FIELD_MAX_DIR2(int width, int height, float *values, int* patchMax
					 )
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){
    	int k = get(i,j,width);
    	
		float max = values[get(normeValue(i + 1, width), j, width)];
		int maxDir = 0;

		float current = values[get(normeValue(i + 1, width), normeValue(j + 1, height), width)];
		if (current > max) {
			max = current;
			maxDir = 45;
		}

		current = values[get(i, normeValue(j + 1, height), width)];
		if (current > max) {
			max = current;
			maxDir = 90;
		}

		current = values[get(normeValue(i - 1, width), normeValue(j + 1, height), width)];
		if (current > max) {
			max = current;
			maxDir = 135;
		}

		current = values[get(normeValue(i - 1, width), j, width)];
		if (current > max) {
			max = current;
			maxDir = 180;
		}

		current = values[get(normeValue(i - 1, width), normeValue(j - 1, height), width)];
		if (current > max) {
			max = current;
			maxDir = 225;
		}

		current = values[get(i, normeValue(j - 1, height), width)];
		if (current > max) {
			max = current;
			maxDir = 270;
		}

		current = values[get(normeValue(i + 1, width), normeValue(j - 1, height), width)];
		if (current > max) {
			max = current;
			maxDir = 315;
		}

	   patchMax[k] = maxDir;
	   }
	   
}

//with fields
extern "C"
__global__ void DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIRV2( int width, int height, float *values, float* tmp, float evapCoef, int* patchMax)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){
    	
		float max = tmp[get(normeValue(i + 1, width), j, width)];
		float total = max;
		int maxDir = 0;

		float current = tmp[get(normeValue(i + 1, width), normeValue(j + 1, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 45;
		}

		current = tmp[get(i, normeValue(j + 1, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 90;
		}

		current = tmp[get(normeValue(i - 1, width), normeValue(j + 1, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 135;
		}

		current = tmp[get(normeValue(i - 1, width), j, width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 180;
		}

		current = tmp[get(normeValue(i - 1, width), normeValue(j - 1, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 225;
		}

		current = tmp[get(i, normeValue(j - 1, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 270;
		}

		current = tmp[get(normeValue(i + 1, width), normeValue(j - 1, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 315;
		}

		int k = get(i,j,width);
	   	patchMax[k] = maxDir;
		total += values[k];
	    values[k] = total - total * evapCoef;
	   }
	   
}

//with fields
extern "C"
__global__ void DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIRV3( int width, int height, float *values, float* tmp, float evapCoef, int* patchMax)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
        
       if (i < width && j < height ){
    	
		int iPlusOne = i + 1;
		int jPlusOne = j + 1;
	 	int iMinusOne = i - 1;
	 	int jMinusOne = j - 1;

		float max = tmp[get(normeValue(iPlusOne, width), j, width)];
		float total = max;
		int maxDir = 0;

		float current = tmp[get(normeValue(iPlusOne, width), normeValue(jPlusOne, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 45;
		}

		current = tmp[get(i, normeValue(jPlusOne, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 90;
		}

		current = tmp[get(normeValue(iMinusOne, width), normeValue(jPlusOne, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 135;
		}

		current = tmp[get(normeValue(iMinusOne, width), j, width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 180;
		}

		current = tmp[get(normeValue(iMinusOne, width), normeValue(jMinusOne, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 225;
		}

		current = tmp[get(i, normeValue(jMinusOne, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 270;
		}

		current = tmp[get(normeValue(iPlusOne, width), normeValue(jMinusOne, height), width)];
		total += current;
		if (current > max) {
			max = current;
			maxDir = 315;
		}

		int k = get(i,j,width);
	   	patchMax[k] = maxDir;
		total += values[k];
	    values[k] = total - total * evapCoef;
	   }
	   
}

extern "C"
__global__ void DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIR( int width, int height, float *values, float* tmp, float evapCoef, int* patchMax)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	
       if (i < width && j < height ){//TODO + alone
		int k = get(i,j,width);
		int* neighbors = neighborsIndexes(i,j,width,height);
		int maxIndex = 0;
		float total = tmp[neighbors[0]];
		float max = total;
		for(int u=1 ; u < 8 ; u++){
			float current = tmp[neighbors[u]];
			total += current;
			if(max < current){
				max = current;
				maxIndex = u;
			}
		}
		patchMax[k] = maxIndex * 45;
		total += values[k];
	    	values[k] = total - total * evapCoef;
    	}
}
