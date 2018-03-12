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

public class GPUSobelGradientsPhero extends CudaPheromone {

    private IntBuffer fieldMaxDir;

    protected CUdeviceptr fieldMaxDirPtr;
    protected Pointer maxPinnedMemory;
    private CudaKernel sobel, sobel2;
    private Pointer fieldMaxDirDataGridPtr;

    public GPUSobelGradientsPhero(String name, int width, int height, final float evapCoeff, final float diffCoeff) {
	super(name, width, height, evapCoeff, diffCoeff);
	fieldMaxDirPtr = new CUdeviceptr();
	maxPinnedMemory = new Pointer();
	fieldMaxDir = (IntBuffer) getUnifiedBufferBetweenPointer(maxPinnedMemory, fieldMaxDirPtr, Integer.class);
	fieldMaxDirDataGridPtr = Pointer.to(maxPinnedMemory);
//	sobel = getCudaKernel("SOBEL", "/turtlekit/cuda/kernels/SobelGradient_2D.cu", getKernelConfiguration());
	sobel2 = getCudaKernel("DIFFUSION_UPDATE_AND_SOBEL_THEN_EVAPORATION", "/turtlekit/cuda/kernels/SobelGradient_2D.cu", getKernelConfiguration());
    }

//    /**
//     * This is faster than calling them sequentially: Only one GPU kernel is called.
//     */
//    @Override
//    public void diffusionAndEvaporation() {
//	super.diffusionAndEvaporation();
//	sobel.run(
//		widthPtr, 
//		heightPtr, 
//		dataGridPtr, 
//		fieldMaxDirDataGridPtr);
//    }
    
    @Override
    public void diffusionAndEvaporationUpdateKernel() {
	sobel2.run(
		widthPtr,
		heightPtr,
		dataGridPtr,
		tmpDeviceDataGridPtr,
		getPointerToFloat(getEvaporationCoefficient()),
		fieldMaxDirDataGridPtr);
    }

    @Override
    public int getMaxDirection(int i, int j) {
	return fieldMaxDir.get(get1DIndex(i, j));
    }

    @Override
    public int getMinDirection(int i, int j) {
	return getMaxDirection(i, j) + 180;
    }

    public void freeMemory() {
	super.freeMemory();
	freeCudaMemory(maxPinnedMemory);
	freeCudaMemory(fieldMaxDirPtr);
    }

}
