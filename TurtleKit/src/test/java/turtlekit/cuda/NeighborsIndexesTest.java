package turtlekit.cuda;

import java.io.File;
import java.util.logging.Level;

import org.testng.SkipException;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;

public class NeighborsIndexesTest {
	private CudaEngine cudaEngine;
	private File targetFile;

	private boolean cudaAvailable;

	@BeforeClass
	public void setUp() throws Exception {
		System.setProperty("java.library.path", "/tmp/tklib/lib");
		try {
			CudaPlatform.init(Level.ALL);
			cudaAvailable = true;
		} catch (Throwable e) {
			e.printStackTrace();
			cudaAvailable = false;
		}
	}

	@BeforeMethod
	protected void checkCuda() {
		if (!cudaAvailable) {
			throw new SkipException("Skipping tests because Cuda was not available.");
		}
	}

	@Test
	public void populate() {
		int width = 2;
		NeighborsIndexes neighbors = new NeighborsIndexes(width, width);

		// Allocate memory on the device
		int numElements = width * width * 8;
		int[] hostArray = new int[numElements];
		Pointer devicePtr = neighbors.getNeighborsIndexesPtr();

		// Copy data from device to host
		Pointer hostPtr = new Pointer();
		JCuda.cudaMallocHost(hostPtr, numElements * Sizeof.INT); // Allocate pinned memory on host
		JCuda.cudaMemcpy(hostPtr, devicePtr, numElements * Sizeof.INT, cudaMemcpyKind.cudaMemcpyDeviceToHost);

		// Copy data from pinned host memory to regular host array
		JCuda.cudaMemcpy(Pointer.to(hostArray), hostPtr, numElements * Sizeof.INT, cudaMemcpyKind.cudaMemcpyHostToHost);

		// Free device memory
		JCuda.cudaFree(devicePtr);
		JCuda.cudaFreeHost(hostPtr);

		// Print the results
		for (int i = 0; i < numElements; i++) {
			System.out.println("Element " + i + ": " + hostArray[i]);
		}
	}
}
