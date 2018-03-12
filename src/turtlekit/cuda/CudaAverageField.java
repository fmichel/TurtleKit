package turtlekit.cuda;

import java.nio.FloatBuffer;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

public class CudaAverageField implements CudaObject {
	//TODO ADD UN TABLEAU POUR LES RESULTATS !!!!!!!!!
	final int width, height;
	
	public final static int DEFAULT_DEPTH = 5;
	
	private FloatBuffer values;
	private FloatBuffer result;
	
	CUdeviceptr valuesPtr;
	CUdeviceptr resultPtr;
	
	CudaKernel averageComputation;
	private Pointer valuesPinnedMemory;
	private Pointer resultPinnedMemory;

	private KernelConfiguration kernelConfiguration;

	private Pointer widthPtr;
	private Pointer heightPtr;
	private Pointer dataGridPtr;
	private Pointer resultGridPtr;
	private int defaultDepth;
	
	public CudaAverageField(String name, int width, int height, int depth, float[] valuesAverage) {
		widthPtr = getPointerToInt(width);
		heightPtr = getPointerToInt(height);
		defaultDepth = depth;
		this.width = width;
		this.height = height;
		
		valuesPtr = new CUdeviceptr();
		valuesPinnedMemory = new Pointer();
		values = (FloatBuffer) getUnifiedBufferBetweenPointer(valuesPinnedMemory, valuesPtr, Float.class);
		dataGridPtr = Pointer.to(valuesPinnedMemory);

		resultPtr = new CUdeviceptr();
		resultPinnedMemory = new Pointer();
		result = (FloatBuffer) getUnifiedBufferBetweenPointer(resultPinnedMemory, resultPtr,Float.class);
		resultGridPtr = Pointer.to(resultPinnedMemory);

		initFunctions();
	}
	
	public CudaAverageField(String name, int width, int height, float[] valuesAverage) {
		this(name,width,height,DEFAULT_DEPTH,valuesAverage);
	}
	
	private void initValues(float[] valuesAverage) {//TODO
		values.rewind();
		result.rewind();
		for (float f : valuesAverage) {
			values.put(f);
			result.put(0);
		}
	}
	
	/**
	 * @return the width
	 */
	public int getWidth() {
		return width;
	}

	/**
	 * @return the height
	 */
	public int getHeight() {
		return height;
	}
	
	/**
	 * @return the values
	 */
	public FloatBuffer getValues() {
		return values;
	}

	/**
	 * @param values the values to set
	 */
	public void setValues(FloatBuffer values) {
		this.values = values;
	}
	
	public float get(int index) {
		return values.get(index);
	}
	
	public float getResult(int index) {
		return result.get(index);
	}

	public void set(int index, float value) {
		values.put(index, value);
	}
	
	protected void initFunctions() {
		kernelConfiguration = getNewKernelConfiguration();
		averageComputation = getCudaKernel("AVERAGE_DEPTH_1D_V2", "/turtlekit/cuda/kernels/Average_2D.cu", kernelConfiguration);
	}
	

	public void computeAverage(final int depth){
		averageComputation.run(
				widthPtr,
				heightPtr,
				dataGridPtr,
				resultGridPtr,
				getPointerToInt(depth)
				);
	}
	
	public void computeAverage(){
		computeAverage(defaultDepth);
	}
	
	@Override
	public void freeMemory() {
		freeCudaMemory(valuesPtr);
		freeCudaMemory(resultPtr);
	}

}
