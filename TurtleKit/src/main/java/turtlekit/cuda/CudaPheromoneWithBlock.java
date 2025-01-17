package turtlekit.cuda;

import turtlekit.kernel.TKEnvironment;

public class CudaPheromoneWithBlock extends CudaPheromone{

	public CudaPheromoneWithBlock(String name, TKEnvironment<?> env, float evapPercentage, float diffPercentage) {
		super(name, env, evapPercentage, diffPercentage);
    }


	protected void initKernels() {
//		kernelConfiguration.setStreamID(cudaEngine.getNewCudaStream());
		diffusionToTmpKernel = createKernel("DIFFUSION_TO_TMP", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
		diffusionUpdateKernel = createKernel("DIFFUSION_UPDATE", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
		diffusionUpdateThenEvaporationKernel = createKernel("DIFFUSION_UPDATE_THEN_EVAPORATION", "/turtlekit/cuda/kernels/DiffusionEvaporationWithBlock_2D.cu");
		evaporationKernel = createKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu");
	}
	
	
}
