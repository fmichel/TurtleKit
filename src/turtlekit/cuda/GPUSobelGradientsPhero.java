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

public class GPUSobelGradientsPhero extends CudaPheromone {

    private CudaKernel sobel, sobel2;

    private CudaIntBuffer cudaIntBuffer;

    public GPUSobelGradientsPhero(String name, int width, int height, final float evapCoeff, final float diffCoeff) {
	super(name, width, height, evapCoeff, diffCoeff);
//	sobel = getCudaKernel("SOBEL", "/turtlekit/cuda/kernels/SobelGradient_2D.cu", getKernelConfiguration());
	sobel2 = createKernel("DIFFUSION_UPDATE_AND_SOBEL_THEN_EVAPORATION", "/turtlekit/cuda/kernels/SobelGradient_2D.cu");
	cudaIntBuffer = new CudaIntBuffer(this);
    }

    @Override
    public void diffusionAndEvaporationUpdateKernel() {
	sobel2.run(
		getWidthPointer(),
		getHeightPointer(),
		getValues().getPointer(),
		tmpDeviceDataGridPtr,
		getPointerToFloat(getEvaporationCoefficient()),
		cudaIntBuffer.getDataPpointer());
    }

    @Override
    public int getMaxDirection(int i, int j) {
	return cudaIntBuffer.get(get1DIndex(i, j));
    }

    @Override
    public int getMinDirection(int i, int j) {
	return getMaxDirection(i, j) + 180;
    }

    public void freeMemory() {
	super.freeMemory();
	cudaIntBuffer.freeMemory();
    }

}
