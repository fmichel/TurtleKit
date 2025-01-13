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

public class CudaPheromoneWithBlock extends CudaPheromone{
	

	public CudaPheromoneWithBlock(String name, int width, int height, float evapPercentage, float diffPercentage) {
	    super(name, width, height, evapPercentage, diffPercentage);
    }
	
	public CudaPheromoneWithBlock(String name, int width, int height, final int evapPercentage, final int diffPercentage) {
	    super(name, width, height, evapPercentage / 100f, diffPercentage / 100f);
}


	protected void initKernels() {
//		kernelConfiguration.setStreamID(cudaEngine.getNewCudaStream());
		diffusionToTmpKernel = createKernel("DIFFUSION_TO_TMP", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
		diffusionUpdateKernel = createKernel("DIFFUSION_UPDATE", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
		diffusionUpdateThenEvaporationKernel = createKernel("DIFFUSION_UPDATE_THEN_EVAPORATION", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
		evaporationKernel = createKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu");
	}
	
	
}
