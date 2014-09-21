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
import jcuda.driver.CUfunction;
import jcuda.driver.CUmemorytype;
import jcuda.driver.CUstream;
import jcuda.driver.JCudaDriver;
import turtlekit.cuda.CudaEngine.Kernel;

public class CudaPheromone extends turtlekit.pheromone.Pheromone implements CudaObject{
	
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
	CUdeviceptr tmpPtr;
	private Runnable diffusionUpdate;
	private Pointer valuesPinnedMemory;
	private int gridSizeY;
	private Runnable evaporation;
//	private Runnable diffusionAndEvaporation;
	private Runnable diffusionUpdateThenEvaporation;
	protected Pointer arrPointer;
	private Runnable test;
	protected CUdeviceptr testDevicePtr;
	

	public CudaPheromone(String name, int width, int height, final int evapPercentage,
			final int diffPercentage) {
		this(name, width, height, evapPercentage / 100f, diffPercentage / 100f);
	}
	
	public CudaPheromone(String name, int width, int height, final float evapPercentage,
			final float diffPercentage) {
		super(name, width, height, evapPercentage, diffPercentage);
		widthParam = new int[]{width};
		heightParam = new int[]{height};
//		arr = new float[width * height];
//		Arrays.fill(arr, 0);
		cudaEngine = CudaEngine.getCudaEngine(this);
		initCuda();
		initFunctions();
	}
	
	@Override
	public float get(int index) {
		return values.get(index);
	}

	@Override
	public void set(int index, float value) {
		if(value > getMaximum())
			setMaximum(value);
		values.put(index, value);
	}

	public void initCuda() {
		try {
			initCudaParameters();
			initCudaStream();
			initCudaTmpGrid();
			initCudaValues();
		} catch (InterruptedException | ExecutionException | NullPointerException e) {
			e.printStackTrace();
		}
	}

	public void initCudaValues() throws InterruptedException, ExecutionException {
		final int floatGridMemorySize = getWidth() * getHeight() * Sizeof.FLOAT;
			cudaEngine.submit(new Runnable() {
				public void run() {
					valuesPtr = new CUdeviceptr();
					valuesPinnedMemory = new Pointer();
					values = CudaEngine.getUnifiedFloatBuffer(valuesPinnedMemory,
							valuesPtr, floatGridMemorySize);
				}
			}).get();
	}

	public void initCudaTmpGrid() throws InterruptedException, ExecutionException {
		final int floatGridMemorySize = getWidth() * getHeight() * Sizeof.FLOAT;
			cudaEngine.submit(new Runnable() {
				public void run() {
					tmpPtr = new CUdeviceptr();
					cuMemAlloc(tmpPtr, floatGridMemorySize);
				}
			}).get();
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
					
					tmpPtr = new CUdeviceptr();
					cuMemAlloc(tmpPtr, floatGridMemorySize);
				}
			}).get();
	}

	public void initCudaStream() throws InterruptedException, ExecutionException {
			cudaEngine.submit(new Runnable() {
				public void run() {
					cudaStream = new CUstream();
					JCudaDriver.cuStreamCreate(cudaStream, 0);
				}
			}).get();
	}

	/**
	 * 
	 */
	protected void initCudaParameters() {
		MAX_THREADS = cudaEngine.getMaxThreads();
//		MAX_THREADS = 16;
		gridSizeX = (getWidth() + MAX_THREADS - 1) / MAX_THREADS;
		gridSizeY = (getHeight() + MAX_THREADS - 1) / MAX_THREADS;
	}
	
	public Pointer getParamterPointer(Pointer... kernelParameters){
		return Pointer.to(kernelParameters);
	}
	
	public Pointer getPointerToFloat(float f){
		return Pointer.to(new float[]{f});
	}
	
	protected void initFunctions() {
		final CUfunction diffusionToTmpFunction = cudaEngine.getKernelFunction(Kernel.DIFFUSION_TO_TMP);
		diffusionToTmp = new Runnable() {
			@Override
			public void run() {
				launchKernel(diffusionToTmpFunction,
						Pointer.to(widthParam),
						Pointer.to(heightParam),
						Pointer.to(valuesPinnedMemory),
						Pointer.to(tmpPtr),
						getPointerToFloat(getDiffusionCoefficient())
						);
			}
		};

		final CUfunction diffusionUpdateFunction = cudaEngine.getKernelFunction(Kernel.DIFFUSION_UPDATE);
		diffusionUpdate = new Runnable() {
			@Override
			public void run() {
				launchKernel(diffusionUpdateFunction,
						Pointer.to(widthParam),
						Pointer.to(heightParam),
						Pointer.to(valuesPinnedMemory),
						Pointer.to(tmpPtr)
						);
			}
		};

		final CUfunction diffusionUpdateThenEvaporationFunction = cudaEngine.getKernelFunction(Kernel.DIFFUSION_UPDATE_THEN_EVAPORATION);
		diffusionUpdateThenEvaporation = new Runnable() {
			@Override
			public void run() {
				launchKernel(diffusionUpdateThenEvaporationFunction,
						Pointer.to(widthParam),
						Pointer.to(heightParam),
						Pointer.to(valuesPinnedMemory),
						Pointer.to(tmpPtr),
						getPointerToFloat(getEvaporationCoefficient())
						);
			}
		};

		
		final CUfunction evaporationFunction = cudaEngine.getKernelFunction(Kernel.EVAPORATION);
		evaporation = new Runnable() {
			@Override
			public void run() {
				launchKernel(evaporationFunction, 
						Pointer.to(widthParam),
						Pointer.to(heightParam),
						Pointer.to(valuesPinnedMemory),
						getPointerToFloat(getEvaporationCoefficient())
						);

//				launchKernel(kernelParameters, evaporationFunction);
			}
		};
		
		final CUfunction testFunction = cudaEngine.getKernelFunction(Kernel.TEST);
		test = new Runnable() {
			@Override
			public void run() {
				launchKernel(testFunction, 
						Pointer.to(widthParam),
						Pointer.to(heightParam),
						testDevicePtr
						);
			}
		};
		
}

	protected void launchKernel(CUfunction cUfunction, Pointer... parameters) {
		JCudaDriver.cuLaunchKernel(cUfunction, //TODO cach
				gridSizeX , gridSizeY, 1, // Grid dimension
				MAX_THREADS , MAX_THREADS, 1, // Block dimension
				0, cudaStream, // Shared memory size and stream
				Pointer.to(parameters), null // Kernel- and extra parameters
		);
	}

	public CudaEngine getCudaEngine(){
		return cudaEngine;
	}
	
	public void diffusion(){
		if (getDiffusionCoefficient() != 0) {
			cudaEngine.submit(diffusionToTmp);
			cudaEngine.submit(diffusionUpdate);
		}
	}

	/**
	 * This is faster than calling them sequentially: 
	 * Only one GPU kernel is called.
	 * 
	 */
	@Override
	public void diffusionAndEvaporation() {
		if(getDiffusionCoefficient() != 0 && getEvaporationCoefficient() != 0){
			cudaEngine.submit(diffusionToTmp);
			cudaEngine.submit(diffusionUpdateThenEvaporation);
//			System.err.println("arr "+arr[0]);
//			cudaEngine.submit(test);//TODO
//			System.err.println(arr[1]);
		}
		else{
			diffusion();
			evaporation();
		}
	}
	
	@Override
	public void evaporation() {
		if (getEvaporationCoefficient() != 0) {
			cudaEngine.submit(evaporation);
		}
//		updateFieldMaxDir();
	}
	
//	public void updateFieldMaxDir() {
//			cuda.submit(fieldMinDirComputation);
//	}
	
	public void freeMemory() {
		cudaEngine.submit(new Runnable() {
			@Override
			public void run() {
				cuMemFree(tmpPtr);
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

	@Override
	public FloatBuffer getValuesFloatBuffer() {
		return getValues();
	}

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

}
