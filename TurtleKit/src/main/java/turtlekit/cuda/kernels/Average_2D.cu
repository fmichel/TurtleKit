
// Kernel Average with Depth
extern "C"
__global__ void AVERAGE_DEPTH_1D(int envSizeX, int envSizeY, float* envData, int depth){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	float moyenne = 0;
	int nbNombre = 0;

	if(tidX < envSizeX && tidY < envSizeY){
		for(int l = tidX - depth; l <= tidX + depth; l++){
			if(l < 0){
				int ltemp = l;
				ltemp += envSizeX;

				for(int k = tidY - depth; k <= tidY + depth; k++){
					if(k < 0){
						int ktemp = k;
						ktemp += envSizeY;
						if(envData[envSizeX * ltemp + ktemp] != -1){
							moyenne += envData[envSizeX * ltemp + ktemp];
							nbNombre++;
						}
					}
					else if(k > envSizeY - 1){
						int ktemp = k;
						ktemp -= envSizeY;
						if(envData[envSizeX * ltemp + ktemp] != -1){
							moyenne += envData[envSizeX * ltemp + ktemp];
							nbNombre++;
						}
					}
					else{
						if(envData[envSizeX * ltemp + k] != -1){
							moyenne += envData[envSizeX * ltemp + k];
							nbNombre++;
						}
					}
				}
			}
			else if(l > envSizeX - 1){
				int ltemp = l;
				ltemp -= envSizeX;

				for(int k = tidY - depth; k <= tidY + depth; k++){
					if(k < 0){
						int ktemp = k;
						ktemp += envSizeY;
						if(envData[envSizeX * ltemp + ktemp] != -1){
							moyenne += envData[envSizeX * ltemp + ktemp];
							nbNombre++;
						}
					}
					else if(k > envSizeY - 1){
						int ktemp = k;
						ktemp -= envSizeY;
						if(envData[envSizeX * ltemp + ktemp] != -1){
							moyenne += envData[envSizeX * ltemp + ktemp];
							nbNombre++;
						}
					}
					else{
						if(envData[envSizeX * ltemp + k] != -1){
							moyenne += envData[envSizeX * ltemp + k];
							nbNombre++;
						}
					}
				}
			}
			else{
				for(int k = tidY - depth; k <= tidY + depth; k++){
					if(k < 0){
						int ktemp = k;
						ktemp += envSizeY;
						if(envData[envSizeX * l + ktemp] != -1){
							moyenne += envData[envSizeX * l + ktemp];
							nbNombre++;
						}
					}
					else if(k > envSizeY - 1){
						int ktemp = k;
						ktemp -= envSizeY;
						if(envData[envSizeX * l + ktemp] != -1){
							moyenne += envData[envSizeX * l + ktemp];
							nbNombre++;
						}
					}
					else{
						if(envData[envSizeX * l + k] != -1){
							moyenne += envData[envSizeX * l + k];
							nbNombre++;
						}
					}
				}
			}
		}
		if(nbNombre != 0){
			envData[envSizeX * tidX + tidY] = moyenne / nbNombre;
		}
	}
	__syncthreads();
}

//Converting 2D coordinates into one 1D coordinate
__device__ int getFOV(int x, int y,int width){
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
__global__ void AVERAGE_DEPTH_1D_V2(int envSizeX, int envSizeY, float* envData, float* result, int depth){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	float moyenne = 0;
	float nbNombre = 0;

	if(tidX < envSizeX && tidY < envSizeY){
		int borneInfX = tidX - depth;
		int borneSupX = tidX + depth;
		int borneInfY = tidY - depth;
		int borneSupY = tidY + depth;
		for(int i = borneInfX; i <= borneSupX; i++){
			for(int j = borneInfY; j <= borneSupY; j++){
				float valeur = envData[getFOV(normeFOV(i,envSizeX),normeFOV(j,envSizeY),envSizeY)];
				if(valeur != -1){
					moyenne += valeur;
					nbNombre++;
				}
			}
		}
		if(nbNombre != 0){
			result[envSizeY * tidX + tidY] = moyenne / nbNombre;
		}
	}
}

//Heat Diffusion Kernel
extern "C"
__global__ void HEAT_DEPTH_1D_V2(int envSizeX, int envSizeY, float* envData, float* result, int depth){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;
	
	float actualHeat = envData[getFOV(normeFOV(tidX,envSizeX),normeFOV(tidY,envSizeY),envSizeY)];
	float heat = 0;

	if(tidX < envSizeX && tidY < envSizeY){
		int borneInfX = tidX - depth;
		int borneSupX = tidX + depth;
		int borneInfY = tidY - depth;
		int borneSupY = tidY + depth;
		for(int i = borneInfX; i <= borneSupX; i++){
			for(int j = borneInfY; j <= borneSupY; j++){
				heat += envData[getFOV(normeFOV(i,envSizeX),normeFOV(j,envSizeY),envSizeY)];
			}
		}
		heat -= actualHeat;
		envData[envSizeY * tidY + tidX] = (actualHeat + 0.125f * (heat - 8 * actualHeat));
	}
}

//Number Neighbors Kernel
extern "C"
__global__ void NUMBER_NEIGHBORS_ALIVE(int envSizeX, int envSizeY, float* envData, float* result, int depth){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	float temp = 0.0f;

	if(tidX < envSizeX && tidY < envSizeY){
		int borneInfX = tidX - depth;
		int borneSupX = tidX + depth;
		int borneInfY = tidY - depth;
		int borneSupY = tidY + depth;
		for(int i = borneInfX; i <= borneSupX; i++){
			for(int j = borneInfY; j <= borneSupY; j++){
				if(!(i == tidX && j == tidY)){
					if(envData[getFOV(normeFOV(i,envSizeX),normeFOV(j,envSizeY),envSizeY)] == 1.0f){
						temp++;
					}
				}
			}
		}
		__syncthreads();
		result[envSizeY * tidY + tidX] = temp;
	}
}

//State Computation Kernel
extern "C"
__global__ void STATE_COMPUTATION(int envSizeX, int envSizeY, float* envData, float* result, int depth){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	float temp = 0.0f;

	if(tidX < envSizeX && tidY < envSizeY){
		int borneInfX = tidX - depth;
		int borneSupX = tidX + depth;
		int borneInfY = tidY - depth;
		int borneSupY = tidY + depth;
		for(int i = borneInfX; i <= borneSupX; i++){
			for(int j = borneInfY; j <= borneSupY; j++){
				if(!(i == tidX && j == tidY)){
					if(envData[getFOV(normeFOV(i,envSizeX),normeFOV(j,envSizeY),envSizeY)] == -1.0f){
						temp--;
					}
					if(envData[getFOV(normeFOV(i,envSizeX),normeFOV(j,envSizeY),envSizeY)] == 1.0f){
						temp++;
					}
				}
			}
		}
		__syncthreads();
		result[envSizeY * tidY + tidX] = temp;
	}
}

//Here Computation Kernel
extern "C"
__global__ void HERE_COMPUTATION(int envSizeX, int envSizeY, float* envData, float* result, int depth){
	int tidX = blockIdx.x * blockDim.x + threadIdx.x;
	int tidY = blockIdx.y * blockDim.y + threadIdx.y;

	float temp = 0.0f;

	if(tidX < envSizeX && tidY < envSizeY){
		int borneInfX = tidX - depth;
		int borneSupX = tidX + depth;
		int borneInfY = tidY - depth;
		int borneSupY = tidY + depth;
		for(int i = borneInfX; i <= borneSupX; i++){
			for(int j = borneInfY; j <= borneSupY; j++){
				if(!(i == tidX && j == tidY)){
					if(envData[getFOV(normeFOV(i,envSizeX),normeFOV(j,envSizeY),envSizeY)] == 1.0f){
						temp++;
					}
				}
			}
		}
		__syncthreads();
		result[envSizeY * tidY + tidX] = temp;
	}
}
