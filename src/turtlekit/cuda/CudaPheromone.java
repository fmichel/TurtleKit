/*******************************************************************************
 * TurtleKit 3 - Agent Based and Artificial Life Simulation Platform
 * Copyright (C) 2011-2014 Fabien Michel
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package turtlekit.cuda;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemFreeHost;

import java.nio.FloatBuffer;
import java.util.concurrent.ExecutionException;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUDA_ARRAY_DESCRIPTOR;
import jcuda.driver.CUDA_MEMCPY2D;
import jcuda.driver.CUarray;
import jcuda.driver.CUarray_format;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import turtlekit.pheromone.AbstractPheromoneGrid;
import turtlekit.pheromone.Pheromone;

public class CudaPheromone extends AbstractPheromoneGrid<Float> implements CudaObject,Pheromone<Float>{
	
	private FloatBuffer values;
//	private float[] arr;
	
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

	private CudaEngine cudaEngine;
	CUdeviceptr valuesPtr;
	private CUstream cudaStream;
	int[] widthParam;
	int[] heightParam;
	private int MAX_THREADS;
	private int gridSizeX;
	Runnable diffusionToTmp;
	CUdeviceptr tmpDeviceDataGrid;
	private Runnable diffusionUpdate;
	private Pointer valuesPinnedMemory;
	private int gridSizeY;
	private Runnable evaporation;
//	private Runnable diffusionAndEvaporation;
	private Runnable diffusionUpdateThenEvaporation;
	protected Pointer arrPointer;
	private Runnable test;
	protected CUdeviceptr testDevicePtr;
	protected Pointer heightPtr;
	protected Pointer dataGridPtr;
	private turtlekit.cuda.CudaKernel diffusionToTmpKernel;
	private CudaKernel diffusionUpdateKernel;
	private CudaKernel diffusionUpdateThenEvaporationKernel;
	private CudaKernel evaporationKernel;
	protected Pointer widthPtr;
	private KernelConfiguration kernelConfiguration;
	protected Pointer tmpDeviceDataGridPtr;
	

	public KernelConfiguration getKernelConfiguration() {
		return kernelConfiguration;
	}

	public void setKernelConfiguration(KernelConfiguration kernelConfiguration) {
		this.kernelConfiguration = kernelConfiguration;
	}

	public CudaPheromone(String name, int width, int height, final int evapPercentage,
			final int diffPercentage) {
		this(name, width, height, evapPercentage / 100f, diffPercentage / 100f);
	}
	
	public CudaPheromone(String name, int width, int height, final float evapPercentage,
			final float diffPercentage) {
		super(name, width, height, evapPercentage, diffPercentage);
		setMaximum(0f);
		widthParam = new int[]{width};
		widthPtr = Pointer.to(new int[]{width});
		heightParam = new int[]{height};
		heightPtr = Pointer.to(new int[]{height});
//		arr = new float[width * height];
//		Arrays.fill(arr, 0);
		cudaEngine = CudaEngine.getCudaEngine(this);
		tmpDeviceDataGrid = cudaEngine.createDeviceDataGrid(getWidth(), getHeight(), Float.class);
		tmpDeviceDataGridPtr = Pointer.to(tmpDeviceDataGrid);
		valuesPtr = new CUdeviceptr();
		valuesPinnedMemory = new Pointer();
		values = (FloatBuffer) cudaEngine.getUnifiedBufferBetweenPointer(valuesPinnedMemory, valuesPtr, Float.class, getWidth(), getHeight());
		dataGridPtr = Pointer.to(valuesPinnedMemory);

//		initCuda();
		initFunctions();

	}
	
	@Override
	public Float get(int index) {
		return values.get(index);
	}

	@Override
	public void set(int index, Float value) {
		if(value > getMaximum())
			setMaximum(value);
		values.put(index, value);
	}

	public void initCudaTmpGrid() throws InterruptedException, ExecutionException {
//		tmpPtr = cudaEngine.getDevicePointerToDataGrid(getWidth(), getHeight(), Float.class);
//		final int floatGridMemorySize = getWidth() * getHeight() * Sizeof.FLOAT;
//		tmpPtr = new CUdeviceptr();
//			cudaEngine.submit(new Runnable() {
//				public void run() {
//					cuMemAlloc(tmpPtr, floatGridMemorySize);
//				}
//			}).get();
	}

	public void initCudaArray() throws InterruptedException, ExecutionException {
		final int floatGridMemorySize = getWidth() * getHeight() * Sizeof.FLOAT;
			cudaEngine.submit(new Runnable() {
				public void run() {
			        // Create the 2D array on the device
			        CUarray array = new CUarray();
			        CUDA_ARRAY_DESCRIPTOR ad = new CUDA_ARRAY_DESCRIPTOR();
			        ad.Format = CUarray_format.CU_AD_FORMAT_FLOAT;
			        ad.Width = getWidth();
			        ad.Height = getHeight();
			        ad.NumChannels = 1;
			        JCudaDriver.cuArrayCreate(array, ad);
			       
			        // Copy the host input to the 2D array  
			        CUDA_MEMCPY2D copyHD = new CUDA_MEMCPY2D();
			        copyHD.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_UNIFIED;
//			        copyHD.srcHost = Pointer.to(arr);

			        copyHD.srcPitch = getWidth() * Sizeof.FLOAT;
			        copyHD.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_UNIFIED;
			        copyHD.dstArray = array;
			        copyHD.WidthInBytes = getWidth() * Sizeof.FLOAT;
			        copyHD.Height = getHeight();
			        JCudaDriver.cuMemcpy2D(copyHD);
			 
			        testDevicePtr = new CUdeviceptr();
					JCudaDriver.cuMemHostGetDevicePointer(testDevicePtr, arrPointer, 0);
					
					tmpDeviceDataGrid = new CUdeviceptr();
					cuMemAlloc(tmpDeviceDataGrid, floatGridMemorySize);
				}
			}).get();
	}
//
//	public void initCudaStream() throws InterruptedException, ExecutionException {
//			cudaEngine.submit(new Runnable() {
//				public void run() {
//					cudaStream = new CUstream();
//					JCudaDriver.cuStreamCreate(cudaStream, CUstream_flags.CU_STREAM_NON_BLOCKING);
//				}
//			}).get();
//	}

	/**
	 * 
	 */
	public Pointer getPointerToFloat(float f){
		return Pointer.to(new float[]{f});
	}
	
	
	
	protected void initFunctions() {
		kernelConfiguration = cudaEngine.getDefaultKernelConfiguration(getWidth(), getHeight());
//		kernelConfiguration.setStreamID(cudaEngine.getNewCudaStream());
		diffusionToTmpKernel = cudaEngine.getKernel("DIFFUSION_TO_TMP", "/turtlekit/cuda/kernels/Diffusion_2D.cu", kernelConfiguration);
		diffusionUpdateKernel = cudaEngine.getKernel("DIFFUSION_UPDATE", "/turtlekit/cuda/kernels/Diffusion_2D.cu", kernelConfiguration);
		diffusionUpdateThenEvaporationKernel = cudaEngine.getKernel("DIFFUSION_UPDATE_THEN_EVAPORATION", "/turtlekit/cuda/kernels/DiffusionEvaporation_2D.cu", kernelConfiguration);
		evaporationKernel = cudaEngine.getKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu", kernelConfiguration);
}

	public CudaEngine getCudaEngine(){
		return cudaEngine;
	}
	
	protected void diffuseValuesToTmpGridKernel(){
		diffusionToTmpKernel.run(widthPtr, heightPtr, dataGridPtr, tmpDeviceDataGridPtr, getPointerToFloat(getDiffusionCoefficient()));
	}
	
	@Override
	protected void diffusionUpdateKernel() {
		diffusionUpdateKernel.run(widthPtr,
				heightPtr,
				dataGridPtr,
				tmpDeviceDataGridPtr);
	}

	@Override
	public void diffusionAndEvaporationUpdateKernel() {
		diffusionUpdateThenEvaporationKernel.run(widthPtr,
				heightPtr,
				dataGridPtr,
				tmpDeviceDataGridPtr,
				getPointerToFloat(getEvaporationCoefficient()));
	}
	
	@Override
	public void evaporationKernel() {
		evaporationKernel.run(widthPtr, heightPtr,	dataGridPtr, getPointerToFloat(getEvaporationCoefficient()));
	}
	
//	public void updateFieldMaxDir() {
//			cuda.submit(fieldMinDirComputation);
//	}
	
	public void freeMemory() {
		cudaEngine.submit(new Runnable() {
			@Override
			public void run() {
				cuMemFree(tmpDeviceDataGrid);
				cuMemFreeHost(valuesPinnedMemory);
				cuMemFreeHost(valuesPtr);
			}
		});
	}

//	void launchKernel(Pointer kernelParameters, CUfunction cUfunction) {
//		JCudaDriver.cuLaunchKernel(cUfunction, //TODO cach
//				gridSizeX , gridSizeY, 1, // Grid dimension
//				MAX_THREADS , MAX_THREADS, 1, // Block dimension
//				0, cudaStream, // Shared memory size and stream
//				kernelParameters, null // Kernel- and extra parameters
//		);
//	}


	/**
	 * @param cudaEngine the cudaEngine to set
	 */
	public final void setCudaEngine(CudaEngine cudaEngine) {
		this.cudaEngine = cudaEngine;
	}

	/**
	 * @return the valuesPtr
	 */
	public CUdeviceptr getValuesPtr() {
		return valuesPtr;
	}

	/**
	 * @param valuesPtr the valuesPtr to set
	 */
	public void setValuesPtr(CUdeviceptr valuesPtr) {
		this.valuesPtr = valuesPtr;
	}

	/**
	 * @return the valuesPinnedMemory
	 */
	public Pointer getValuesPinnedMemory() {
		return valuesPinnedMemory;
	}

	/**
	 * @param valuesPinnedMemory the valuesPinnedMemory to set
	 */
	public void setValuesPinnedMemory(Pointer valuesPinnedMemory) {
		this.valuesPinnedMemory = valuesPinnedMemory;
	}

	@Override
	public void incValue(int x, int y, float quantity) {
			incValue(get1DIndex(x, y), quantity);
		}

	/**
	 * Adds <code>inc</code> to the current value of the cell 
	 * with the corresponding index
	 * 
	 * @param index cell's index
	 * @param inc how much to add
	 */
	public void incValue(int index, float inc) {
//		inc += get(index);
//		if (inc > maximum)
//			setMaximum(inc);
//		set(index, inc);
		set(index, inc + get(index));
	}


public int getMaxDirection(int xcor, int ycor) {
		float max = get(normeValue(xcor + 1, getWidth()), ycor);
		int maxDir = 0;

		float current = get(normeValue(xcor + 1, getWidth()), normeValue(ycor + 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 45;
		}

		current = get(xcor, normeValue(ycor + 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 90;
		}

		current = get(normeValue(xcor - 1, getWidth()), normeValue(ycor + 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 135;
		}

		current = get(normeValue(xcor - 1, getWidth()), ycor);
		if (current > max) {
			max = current;
			maxDir = 180;
		}

		current = get(normeValue(xcor - 1, getWidth()), normeValue(ycor - 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 225;
		}

		current = get(xcor, normeValue(ycor - 1, getHeight()));
		if (current > max) {
			max = current;
			maxDir = 270;
		}

		current = get(normeValue(xcor + 1, getWidth()), normeValue(ycor - 1, getHeight()));
		if (current > max) {

			max = current;
			maxDir = 315;
		}
		return maxDir;
	}

	public int getMinDirection(int i, int j) {
		float min = get(normeValue(i + 1, getWidth()), j);
		int minDir = 0;

		float current = get(normeValue(i + 1, getWidth()), normeValue(j + 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 45;
		}

		current = get(i, normeValue(j + 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 90;
		}

		current = get(normeValue(i - 1, getWidth()), normeValue(j + 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 135;
		}

		current = get(normeValue(i - 1, getWidth()), j);
		if (current < min) {
			min = current;
			minDir = 180;
		}

		current = get(normeValue(i - 1, getWidth()), normeValue(j - 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 225;
		}

		current = get(i, normeValue(j - 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 270;
		}

		current = get(normeValue(i + 1, getWidth()), normeValue(j - 1, getHeight()));
		if (current < min) {
			min = current;
			minDir = 315;
		}
		return minDir;
	}

}
