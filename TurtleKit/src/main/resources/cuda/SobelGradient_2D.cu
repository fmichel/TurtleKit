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

extern "C"
__global__ void SOBEL( int width, int height, float *values, int* anglesToMaxGradient)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
    
   if (i < width && j < height ){

	int iPlusOne = i + 1;
	int jPlusOne = j + 1;
	int iMinusOne = i - 1;
	int jMinusOne = j - 1;
	
    float e = values[get(normeValue(iPlusOne,width), j, width)] ;
    float ne = values[get(normeValue(iPlusOne,width), normeValue(jPlusOne,height),width)] ;
	float n = values[get(i, normeValue(jPlusOne,height),width)] ;
    float nw = values[get(normeValue(iMinusOne,width), normeValue(jPlusOne,height),width)] ;
    float w = values[get(normeValue(iMinusOne,width), j, width)] ;
    float sw = values[get(normeValue(iMinusOne,width), normeValue(jMinusOne,height),width)] ;
    float s = values[get(i, normeValue(jMinusOne,height),width)] ;
    float se = values[get(normeValue(iPlusOne,width), normeValue(jMinusOne,height),width)];
	
	
	anglesToMaxGradient[get(i,j,width)] =
		(int) (57.29577951 * atan2(
			nw + 2 * n + ne - se - 2 * s - sw, // filter Y
			ne + 2 * e + se - sw - 2 * w - nw// filter X
		));
	}
}



__device__ float getTotalUpdateFromNeighborsAndComputeSobel(float* tmp, int i, int j, int width, int height, int* gradientAngles){

	int iPlusOne = i + 1;
	int jPlusOne = j + 1;
	int iMinusOne = i - 1;
	int jMinusOne = j - 1;
	
    float e = tmp[get(normeValue(iPlusOne, width), j, width)] ;
    float ne = tmp[get(normeValue(iPlusOne,width), normeValue(jPlusOne,height),width)] ;
	float n = tmp[get(i, normeValue(jPlusOne,height),width)] ;
    float nw = tmp[get(normeValue(iMinusOne,width), normeValue(jPlusOne,height),width)] ;
    float w = tmp[get(normeValue(iMinusOne,width), j, width)] ;
    float sw = tmp[get(normeValue(iMinusOne,width), normeValue(jMinusOne,height),width)] ;
    float s = tmp[get(i, normeValue(jMinusOne,height),width)] ;
    float se = tmp[get(normeValue(iPlusOne,width), normeValue(jMinusOne,height),width)];
	
	gradientAngles[get(i,j,width)] =
		__double2int_rd(57.29577951 * atan2(
			nw + 2 * n + ne - se - 2 * s - sw, // filter Y
			ne + 2 * e + se - sw - 2 * w - nw  // filter X
		));

	return e + ne + n + nw + w + sw + s + se;
}



extern "C"
__global__ void DIFFUSION_UPDATE_AND_SOBEL_THEN_EVAPORATION( int width, int height, float *values, float* tmp, float evapCoef, int* gradientAngles)
{
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        
       if (i < width && j < height ){//TODO + alone
        int k = get(i,j,width);
        float total = values[k] + getTotalUpdateFromNeighborsAndComputeSobel(tmp, i, j, width, height, gradientAngles);
    	values[k] = total - total * evapCoef;
    	}
}


