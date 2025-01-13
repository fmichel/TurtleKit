package turtlekit.cuda;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

public class NeighborsIndexes implements CudaObject {

	
	private int width;
	private int height;
	private KernelConfiguration kernelConfiguration;
	protected CudaKernel computeNeighbors;
	private Pointer neighborsIndexesPtr;
	private CUdeviceptr neighborsIndexes;

	public NeighborsIndexes(int width, int height) {
		this.width = width;
		this.height = height;
		neighborsIndexes = createDeviceDataGrid(width * height * 8, int.class);
		neighborsIndexesPtr = Pointer.to(neighborsIndexes);
		initKernels();
		populateIndexes();
	}
	
	private void populateIndexes() {
		computeNeighbors.run(
				getWidthPointer(),
				getHeightPointer(),
				neighborsIndexesPtr,
				getPointerToInt(getWidth()*getHeight()*8)
				);
	}

	protected void initKernels() {
		kernelConfiguration = createDefaultKernelConfiguration();
		computeNeighbors = createKernel("computeNeighborsIndexes", "neighbors_indexes");
	}
	
	@Override
	public void freeMemory() {
		freeCudaMemory(neighborsIndexes);
	}

	@Override
	public int getWidth() {
		return width;
	}

	@Override
	public int getHeight() {
		return height;
	}

	@Override
	public KernelConfiguration getKernelConfiguration() {
		return kernelConfiguration;
	}

	public Pointer getNeighborsIndexesPtr() {
		return neighborsIndexesPtr;
	}

}
