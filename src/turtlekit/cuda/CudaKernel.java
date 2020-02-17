package turtlekit.cuda;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.logging.Level;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;

public class CudaKernel {
	
	private CUfunction myFonction;
	private CudaEngine myCe;
	private KernelConfiguration kc;
	private Pointer[] parameters;
	private Runnable myJob;


	CudaKernel(CUfunction function, CudaEngine ce, String kernelFunctionName, String dotCuSourceFilePath, KernelConfiguration kc) {
		myFonction = function;
		myCe = ce;
		this.kc = kc;
		myJob = new Runnable() {
			@Override
			public void run() {
				JCudaDriver.cuLaunchKernel(myFonction, //TODO cach
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
		myCe.submit(myJob);
	}
	
	
	

	private void compilePtx(String dotCuSourceFilePath) {
		try {
			final URL resource = getClass().getResource(dotCuSourceFilePath);
			if(resource == null)
				throw new FileNotFoundException(dotCuSourceFilePath+" not found on the class path");
			File f = new File(resource.toURI());
			final Path path = Paths.get(CudaEngine.ioTmpDir, f.getName());
			final File file = path.toFile();
			if(! file.exists() || file.lastModified() < f.lastModified()){
				Files.copy(f.toPath(), path);
			}
		} catch (URISyntaxException | IOException e) {
			e.printStackTrace();
		}
		
//		try(InputStream is = CudaEngine.class.getResourceAsStreéam(dotCuSourceFilePath)){
		
	}


	public static void main(String[] args) {
		CudaObject name = new CudaObject() {
			
			@Override
			public void freeMemory() {
				
			}

			@Override
			public int getHeight() {
				// TODO Auto-generated method stub
				return 0;
			}

			@Override
			public int getWidth() {
				// TODO Auto-generated method stub
				return 0;
			}

			@Override
			public KernelConfiguration getKernelConfiguration() {
			    // TODO Auto-generated method stub
			    return null;
			}
		};
		CudaEngine.init(Level.ALL.toString());
		CudaEngine ce = CudaEngine.getCudaEngine(name);
	}

}
