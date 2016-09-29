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

import java.nio.IntBuffer;

import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;

public class CudaGPUGradientsPhero extends CudaPheromone{
	
	private IntBuffer fieldMaxDir;
	
	protected CUdeviceptr fieldMaxDirPtr;
	protected Pointer maxPinnedMemory;
	private CudaKernel diffusionUpdateAndEvaporationAndFieldMaxDirKernel;
	private Pointer fieldMaxDirDataGridPtr;

	public CudaGPUGradientsPhero(String name, int width, int height, final float evapCoeff, final float diffCoeff) {
		super(name, width, height, evapCoeff, diffCoeff);
		fieldMaxDirPtr = new CUdeviceptr();
		maxPinnedMemory = new Pointer();
		fieldMaxDir = (IntBuffer) getUnifiedBufferBetweenPointer(maxPinnedMemory, fieldMaxDirPtr, Integer.class);
		fieldMaxDirDataGridPtr = Pointer.to(maxPinnedMemory);
		diffusionUpdateAndEvaporationAndFieldMaxDirKernel = getCudaKernel("DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIRV2", "/turtlekit/cuda/kernels/DiffusionEvaporationGradients_2D.cu", getKernelConfiguration());
	}
		
	/**
	 * This is faster than calling them sequentially: 
	 * Only one GPU kernel is called.
	 * 
	 */
	@Override
	public void diffusionAndEvaporation() {
		diffuseValuesToTmpGridKernel();
		diffusionUpdateAndEvaporationAndFieldMaxDirKernel.run(
				widthPtr,
				heightPtr,
				dataGridPtr,
				tmpDeviceDataGridPtr,
				getPointerToFloat(getEvaporationCoefficient()),
				fieldMaxDirDataGridPtr
				);
	}
	
	@Override
	public int getMaxDirection(int i, int j) {
		return fieldMaxDir.get(get1DIndex(i, j));
//		return fieldMaxDirAsArray[get1DIndex(i, j)];
	}
	
//	public void updateFieldMaxDir() {
//			cuda.submit(fieldMinDirComputation);
//	}
	
	public void freeMemory() {
		super.freeMemory();
		freeCudaMemory(maxPinnedMemory);
		freeCudaMemory(fieldMaxDirPtr);
	}

}
