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

__device__ void neighborsIndexes(int i, int j, int width, int height, int* neighbors) {
	int iPlusOne = normeValue(i + 1, width);
	int jPlusOne = normeValue(j + 1, height);
 	int iMinusOne = normeValue(i - 1, width);
 	int jMinusOne = normeValue(j - 1, height);
    neighbors[0] = get(iPlusOne, j, width);
    neighbors[1] = get(iPlusOne, jPlusOne, width);
    neighbors[2] = get(i, jPlusOne, width);
    neighbors[3] = get(iMinusOne, jPlusOne, width);
    neighbors[4] = get(iMinusOne, j, width);
    neighbors[5] = get(iMinusOne, jMinusOne, width);
    neighbors[6] = get(i, jMinusOne, width);
    neighbors[7] = get(iPlusOne, jMinusOne, width);
}

extern "C"
__global__ void computeNeighborsIndexes(int width, int height, int* neighbors, int numElements)
{
       int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
    if (idx < numElements) {
        int i = idx / width; // Calculate row index
        int j = idx % width; // Calculate column index

        neighborsIndexes(i, j, width, height, neighbors + idx * 8); // Assuming 8 neighbors
        neighbors[10] = 100;
    }
}

