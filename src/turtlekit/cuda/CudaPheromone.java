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

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import turtlekit.pheromone.AbstractPheromoneGrid;
import turtlekit.pheromone.Pheromone;

public class CudaPheromone extends AbstractPheromoneGrid<Float> implements CudaObject,Pheromone<Float>{
	
	private CudaFloatBuffer values;
//	private float[] arr;
	
	protected CudaKernel diffusionToTmpKernel;
	protected CudaKernel diffusionUpdateKernel;
	protected CudaKernel diffusionUpdateThenEvaporationKernel;
	protected CudaKernel evaporationKernel;
	protected KernelConfiguration kernelConfiguration;
	protected Pointer tmpDeviceDataGridPtr;
	CUdeviceptr tmpDeviceDataGrid;
	

	public KernelConfiguration getKernelConfiguration() {
		return kernelConfiguration;
	}

	public CudaPheromone(String name, int width, int height, final int evapPercentage,
			final int diffPercentage) {
		this(name, width, height, evapPercentage / 100f, diffPercentage / 100f);
	}
	
	public CudaPheromone(String name, int width, int height, final float evapPercentage,
			final float diffPercentage) {
		super(name, width, height, evapPercentage, diffPercentage);
		setMaximum(0f);
		tmpDeviceDataGrid = createDeviceDataGrid(Float.class);
		tmpDeviceDataGridPtr = Pointer.to(tmpDeviceDataGrid);
		values = new CudaFloatBuffer(this);
		kernelConfiguration = createDefaultKernelConfiguration();
		initKernels();
	}
	
	protected void initKernels() {
		diffusionToTmpKernel = createKernel("DIFFUSION_TO_TMP", "/turtlekit/cuda/kernels/Diffusion_2D.cu");
		diffusionUpdateKernel = createKernel("DIFFUSION_UPDATE", "/turtlekit/cuda/kernels/Diffusion_2D.cu");
		diffusionUpdateThenEvaporationKernel = createKernel("DIFFUSION_UPDATE_THEN_EVAPORATION", "/turtlekit/cuda/kernels/DiffusionEvaporation_2D.cu");
		evaporationKernel = createKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu");
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

	protected void diffuseValuesToTmpGridKernel(){
		diffusionToTmpKernel.run(
				getWidthPointer(),
				getHeightPointer(),
				values.getPointer(), 
				tmpDeviceDataGridPtr, 
				getPointerToFloat(getDiffusionCoefficient()));
	}
	
	@Override
	protected void diffusionUpdateKernel() {
		diffusionUpdateKernel.run(
				getWidthPointer(),
				getHeightPointer(),
				values.getPointer(),
				tmpDeviceDataGridPtr);
	}

	@Override
	public void diffusionAndEvaporationUpdateKernel() {
		diffusionUpdateThenEvaporationKernel.run(
				getWidthPointer(),
				getHeightPointer(),
				values.getPointer(),
				tmpDeviceDataGridPtr,
				getPointerToFloat(getEvaporationCoefficient()));
	}
	
	@Override
	public void evaporationKernel() {
		evaporationKernel.run(
				getWidthPointer(),
				getHeightPointer(),
				values.getPointer(),
				getPointerToFloat(getEvaporationCoefficient()));
	}
	
	public void freeMemory() {
		freeCudaMemory(tmpDeviceDataGrid);
		values.freeMemory();
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

	
	/**
	 * @return the values
	 */
	public CudaFloatBuffer getValues() {
	    return values;
	}

}
