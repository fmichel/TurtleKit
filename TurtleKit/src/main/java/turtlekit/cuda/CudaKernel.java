package turtlekit.cuda;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;

public class CudaKernel {
	
	private CUfunction function;
	private CudaEngine engine;
	private KernelConfiguration kc;
	private Pointer[] parameters;
	private Runnable myJob;


	CudaKernel(CUfunction cufunction, CudaEngine ce, String dotCuSourceFilePath, KernelConfiguration kc) {
		function = cufunction;
		engine = ce;
		this.kc = kc;
		myJob = new Runnable() {
			@Override
			public void run() {
				JCudaDriver.cuLaunchKernel(function, //TODO cach
						kc.getGridDimX() , kc.getGridDimY(), 1, // Grid dimension
						kc.getBlockDimX(), kc.getBlockDimY(), 1, // Block dimension
						0, kc.getStreamID(), // Shared memory size and stream
						Pointer.to(parameters), null // Kernel- and extra parameters
						);
			}
		};
		//TODO bench
//		myJob = () -> JCudaDriver.cuLaunchKernel(myFonction, 
//				kc.getGridDimX() , kc.getGridDimY(), 1, // Grid dimension
//				kc.getBlockDimX(), kc.getBlockDimY(), 1, // Block dimension
//				0, kc.getStreamID(), // Shared memory size and stream
//				Pointer.to(parameters), null // Kernel- and extra parameters
//				);

	}
	
	public void run(Pointer... parameters){
		this.parameters = parameters;
		engine.submit(myJob);
	}
	
	public static void main(String[] args) {
		CudaObject name = new CudaObject() {
			
			@Override
			public void freeMemory() {
				
			}

			@Override
			public int getHeight() {
				return 0;
			}

			@Override
			public int getWidth() {
				return 0;
			}

			@Override
			public KernelConfiguration getKernelConfiguration() {
			    // TODO Auto-generated method stub
			    return null;
			}
		};
		CudaEngine ce = CudaEngine.getCudaEngine(name);
	}

}
