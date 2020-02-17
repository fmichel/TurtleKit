package turtlekit.cuda;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import turtlekit.pheromone.DataGrid;

public abstract class CudaAverageField extends DataGrid<Integer> implements CudaObject {
	
	public final static int DEFAULT_DEPTH = 5;
	
	private CudaIntBuffer values;
	
	CudaKernel averageComputation;
	CudaKernel computeAverageToTmpGrid;

	protected Pointer tmpDeviceDataGridPtr;
	CUdeviceptr tmpDeviceDataGrid;
	
	
	private int defaultDepth;
	
	private KernelConfiguration kernelConfiguration;

	public CudaAverageField(String name, int width, int height, int depth) {
	    super("average",width,height);
		defaultDepth = depth;
		
		tmpDeviceDataGrid = createDeviceDataGrid(Float.class);
		tmpDeviceDataGridPtr = Pointer.to(tmpDeviceDataGrid);
		values = new CudaIntBuffer(this);
		kernelConfiguration = createDefaultKernelConfiguration();
		initKernels();
	}
	
	
//	private void initValues(float[] valuesAverage) {//TODO
//		values.rewind();
//		result.rewind();
//		for (float f : valuesAverage) {
//			values.put(f);
//			result.put(0);
//		}
//	}
	
	public KernelConfiguration getKernelConfiguration() {
		return kernelConfiguration;
	}

	public Integer get(int index) {
		return values.get(index);
	}
	

	public void set(int index, int value) {
		values.put(index, value);
	}
	
	protected void initKernels() {
		averageComputation = createKernel("AVERAGE_DEPTH_1D_V2", "/turtlekit/cuda/kernels/Average_2D.cu");
	}
	

	public void computeAverage(final int depth){
		averageComputation.run(
				getWidthPointer(),
				getHeightPointer(),
				values.getDataPpointer(),
				getPointerToInt(depth)
				);
	}
	
	public void computeAverage(){
		computeAverage(defaultDepth);
	}
	
	@Override
	public void freeMemory() {
	    values.freeMemory();
	}

}
