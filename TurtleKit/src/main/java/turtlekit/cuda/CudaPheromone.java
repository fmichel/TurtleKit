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

import java.util.stream.IntStream;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import turtlekit.kernel.TKEnvironment;
import turtlekit.pheromone.AbstractPheromoneGrid;
import turtlekit.pheromone.GradientsCalculator;
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
	private Pointer indexes;
	private CudaKernel ultra;
	

	public KernelConfiguration getKernelConfiguration() {
		return kernelConfiguration;
	}

	public CudaPheromone(String name, TKEnvironment<?> env, int evapPercentage, int diffPercentage) {
		this(name, env, evapPercentage / 100f, diffPercentage / 100f);
	}
	
	public CudaPheromone(String name, TKEnvironment<?> env, float evapPercentage, float diffPercentage) {
		super(name, env, evapPercentage, diffPercentage);
		setMaxEncounteredValue(Float.NEGATIVE_INFINITY);
		tmpDeviceDataGrid = createDeviceDataGrid(getWidth() * getHeight(), Float.class);
		tmpDeviceDataGridPtr = Pointer.to(tmpDeviceDataGrid);
		values = new CudaFloatBuffer(this);
		kernelConfiguration = createDefaultKernelConfiguration();
		
		NeighborsIndexes neighbors = new NeighborsIndexes(getWidth(), getHeight());
		indexes = neighbors.getNeighborsIndexesPtr();
		
		initKernels();
	}
	
	protected void initKernels() {
		diffusionToTmpKernel = createKernel("DIFFUSION_TO_TMP", "Diffusion_2D");
		diffusionUpdateKernel = createKernel("DIFFUSION_UPDATE", "Diffusion_2D");
		diffusionUpdateThenEvaporationKernel = createKernel("DIFFUSION_UPDATE_THEN_EVAPORATION", "DiffusionEvaporation_2D");
		evaporationKernel = createKernel("EVAPORATION", "Evaporation_2D");

		ultra = createKernel("DIFFUSION_TO_TMP", "pheromones");
	}
	
	@Override
	public Float get(int index) {
		return values.get(index);
	}

	@Override
	public void set(int index, Float value) {
		values.put(index, value);
	}

	public void diffuseValuesToTmpGridKernel() {
		diffusionToTmpKernel.run(
				getWidthPointer(),
				getHeightPointer(),
				values.getPointer(), 
				tmpDeviceDataGridPtr, 
				getPointerToFloat(getDiffusionCoefficient()));
	}
	
	@Override
	public void updateValuesFromTmpGridKernel() {
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
		set(index, inc + get(index));
	}


	public int getMaxDirection(int xcor, int ycor) {
		return getMaxDirectionNoRandom(xcor, ycor);
//		return getMaxDirectionRandomWhenEquals(xcor, ycor);
	}

	public int getMinDirection(int xcor, int ycor) {
		return getMinDirectionNoRandom(xcor, ycor);
//		return getMinDirectionRandomWhenEquals(xcor, ycor);
	}

	public int getMaxDirectionRandomWhenEquals(int xcor, int ycor) {
		int index = get1DIndex(xcor, ycor) * 8;
		return (int) GradientsCalculator.getMaxDirectionRandomWhenEquals(this, index);
	}

	public int getMaxDirectionNoRandom(int xcor, int ycor) {
		int index = get1DIndex(xcor, ycor) * 8;
		return (int) GradientsCalculator.getMaxDirectionNoRandom(this, index);
	}

	public int getMinDirectionRandomWhenEquals(int xcor, int ycor) {
		int index = get1DIndex(xcor, ycor) * 8;
		return (int) GradientsCalculator.getMinDirectionRandomWhenEquals(this, index);
	}

	public int getMinDirectionNoRandom(int xcor, int ycor) {
		int index = get1DIndex(xcor, ycor) * 8;
		return (int) GradientsCalculator.getMinDirectionNoRandom(this, index);
	}

	
	/**
	 * @return the values
	 */
	public CudaFloatBuffer getValues() {
	    return values;
	}
	
	@Override
	public String toString() {
		return super.getName();
	}

	@Override
	public void updateMaxValue() {
		double currentMax = IntStream.range(0, values.size()).parallel().mapToDouble(this::get).max().getAsDouble();
		if (currentMax > getMaxEncounteredValue()) {
			setMaxEncounteredValue((float) currentMax);
		}
		setLogMaxValue(Math.log10((float) getMaxEncounteredValue() + 1) / 256);
	}

}
