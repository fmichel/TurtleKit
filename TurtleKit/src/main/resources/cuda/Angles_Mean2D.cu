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

	if(tidX < envSizeX && tidY < envSizeY){
		int borneInfX = tidX - depth;
		int borneSupX = tidX + depth;
		int borneInfY = tidY - depth;
		int borneSupY = tidY + depth;
	    float sum_sin = 0.0f;
    	float sum_cos = 0.0f;
		int number_of_angles = 0;
		for(int i = borneInfX; i <= borneSupX; i++){
			for(int j = borneInfY; j <= borneSupY; j++){
				float angle = envData[get1DCoord(normeFOV(i,envSizeX),normeFOV(j,envSizeY),envSizeX)];
				if(angle != -1){
					number_of_angles++;
					float sin_angle, cos_angle;
					sincosf(angle, &sin_angle, &cos_angle);
					sum_sin += sin_angle;
					sum_cos += cos_angle;
				}
			}
		}
		if(number_of_angles > 0){
			float mean_sin = sum_sin / number_of_angles;
			float mean_cos = sum_cos / number_of_angles;
			result[get1DCoord(tidX,tidY,envSizeX)] = atan2f(mean_sin, mean_cos);
		}
		else{
			result[get1DCoord(tidX,tidY,envSizeX)] = -1;
		}
	}
}
