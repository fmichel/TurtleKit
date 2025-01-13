//Converting 2D coordinates into one 1D coordinate
__device__ int get1DCoord(int x, int y,int width){
  return y * width + x;
}

//Normalize coordinates for infinite world
__device__ int normeFOV(int x, int width){
	if(x < 0)
		return x + width;
	if(x > width - 1)
		return x - width;
	return x;
}

//Average Kernel
extern "C"
__global__ void ANGLES_MEAN_2D(int envSizeX, int envSizeY, float* envData, float* result, int depth){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	float cosinusSum = 0;
	float sinusSum = 0;
	boolean foundSomeone = false;

	if(tidX < envSizeX && tidY < envSizeY){
		int borneInfX = tidX - depth;
		int borneSupX = tidX + depth;
		int borneInfY = tidY - depth;
		int borneSupY = tidY + depth;
		for(int i = borneInfX; i <= borneSupX; i++){
			for(int j = borneInfY; j <= borneSupY; j++){
				float valeur = envData[get1DCoord(normeFOV(i,envSizeX),normeFOV(j,envSizeY),envSizeX)];
				if(valeur != -1){
					radianValue = radians(valeur);
					cosinusSum += cos(radianValue);
					sinusSum += sin(radianValue);
					foundSomeone = true;
				}
			}
		}
		if(foundSomeone)
			result[get1DCoord(tidX,tidY,envSizeX)] = degrees(atan2(sinusSum,cosinusSum));
		}
		else{
			result[get1DCoord(tidX,tidY,envSizeX)] = -1;
		}
}

