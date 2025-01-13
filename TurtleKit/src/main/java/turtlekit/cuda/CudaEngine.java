/*******************************************************************************
 * TurtleKit 3 - Agent Based and Artificial Life Simulation Platform
 * Copyright (C) 2011-2016 Fabien Michel
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

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemFreeHost;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;

public class CudaEngine {

	static final Map<Integer, CudaEngine> cudaEngines = new ConcurrentHashMap<>();

	private static Logger logger;

	private static AtomicInteger cudaObjectID = new AtomicInteger(0);

	private static Map<CudaObject, CudaEngine> engineBinds = new HashMap<>();

	private ExecutorService exe;
	protected CUfunction f;
	private List<CudaObject> cudaObjects = new ArrayList<>();

	private int maxThreads;

	private int Id = -1;

	protected CUcontext context;

	protected CUmodule myModule;

	private Map<String, CUfunction> functions = new HashMap<>();

	static {
		Runtime.getRuntime().addShutdownHook(new Thread(CudaEngine::stop));
	}

	CudaEngine(final int deviceId) {
		logger = Logger.getLogger(CudaEngine.class.getSimpleName());
		logger.setParent(Logger.getLogger("[MADKIT] "));
		logger.setLevel(Level.ALL);
		// mandatory: Only one cuda thread per context
		exe = Executors.newSingleThreadExecutor(r -> {
			Thread thread = new Thread(r);
			thread.setDaemon(true);
			return thread;
		});
		try {
			submit(() -> {
				CUdevice device = new CUdevice();
				JCudaDriver.cuDeviceGet(device, deviceId);
				int[] array = { 0 };
				JCudaDriver.cuDeviceGetAttribute(array, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
				maxThreads = (int) Math.sqrt(array[0]);
				context = new CUcontext();
				JCudaDriver.cuCtxCreate(context, 0, device);
				myModule = new CUmodule();
	
			}).get();
		} catch (InterruptedException e) {
			Thread.currentThread().interrupt();
		} catch (ExecutionException e) {
			e.printStackTrace();
		}
		Id = deviceId;
		cudaEngines.put(deviceId, this);
	}

	public static CudaEngine getCudaEngine(CudaObject co) {
		return engineBinds.computeIfAbsent(co, v -> {
			final int pheroID = cudaObjectID.incrementAndGet();
			final CudaEngine ce = cudaEngines.get(pheroID % CudaPlatform.getAvailableDevicesCount());
			ce.cudaObjects.add(co);
			logger.finer(co + "ID " + pheroID + " getting cuda engine Id " + ce.Id);
			return ce;
		});
	}

	public Pointer getPointerToFloat(float f) {
		return Pointer.to(new float[] { f });
	}

	@Override
	public String toString() {
		return "cudaEngine device #" + Id;
	}

	/**
	 * Creates a new kernel configuration with a new Cuda stream ID according to 2D
	 * data size.
	 * 
	 * @param dataSizeX
	 * @param dataSizeY
	 * @return a kernel configuration with a unique stream ID
	 */
	public KernelConfiguration createNewKernelConfiguration(int dataSizeX, int dataSizeY) {
		int gridSizeX = (dataSizeX + maxThreads - 1) / maxThreads;
		int gridSizeY = (dataSizeY + maxThreads - 1) / maxThreads;
		return new KernelConfiguration(gridSizeX, gridSizeY, maxThreads, maxThreads, getNewCudaStream());
	}

	/**
	 * Creates a new Cuda kernel using a configuration and a kernel name which could
	 * be found in a Cuda source file.
	 * 
	 * @param functionName
	 * @param cuFileName
	 * @param kc                 a kernel configuration.
	 * @return a new Cuda Kernel
	 * 
	 */
	public CudaKernel createKernel(final String functionName, final String cuFileName,
			final KernelConfiguration kc) {
		Objects.requireNonNull(kc);
		try {
			return exe.submit(() -> {
				CUfunction function = functions.computeIfAbsent("" + functionName + cuFileName,
//						k -> updateCuSourceFile(kernelFunctionName, cuSourceFilePath));
				k -> getFunction(functionName, cuFileName));
				return new CudaKernel(function, CudaEngine.this, cuFileName, kc);
			}).get();
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
		return null;
	}

	public <T> CUdeviceptr createDevicePtr(int nbOfElements, Class<T> dataType) {
		CUdeviceptr ptr = new CUdeviceptr();
		try {
			submit(() -> cuMemAlloc(ptr, getRequiredMemorySize(dataType, nbOfElements))).get();
		} catch (InterruptedException | ExecutionException | IllegalArgumentException | SecurityException e) {
			e.printStackTrace();
		}
		return ptr;
	}

	public <B extends Buffer, T> B getUnifiedBufferBetweenPointer(Pointer hostData, CUdeviceptr deviceData,
			Class<T> dataType, int wdith, int height) {
		int size = getRequiredMemorySize(dataType, wdith * height);
		ByteBuffer buffer;
		try {
			buffer = exe.submit(() -> {
				JCudaDriver.cuMemHostAlloc(hostData, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
				final ByteBuffer byteBuffer = hostData.getByteBuffer(0, size);
				byteBuffer.order(ByteOrder.nativeOrder());
				JCudaDriver.cuMemHostGetDevicePointer(deviceData, hostData, 0);
				return byteBuffer;
			}).get();
			String simpleName = dataType.getSimpleName();
			switch (simpleName) {
			case "Integer":
				return (B) buffer.asIntBuffer();
			case "Character":
				return (B) buffer.asCharBuffer();
			case "Float":
				return (B) buffer.asFloatBuffer();
			default:
				throw new UnsupportedOperationException(" CUDA not ready for that -> " + simpleName);

			}
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
		return null;
	}

	private <T> int getRequiredMemorySize(Class<T> dataType, int nbOfElements) {
		String simpleName = dataType.getSimpleName();
		switch (simpleName) {
		case "int":
			dataType = (Class<T>) Integer.class;
			break;
		case "char":
			dataType = (Class<T>) Character.class;
			break;
		default:
			break;
		}
		int gridMemorySize = 0;
		try {
			gridMemorySize = nbOfElements * dataType.getField("SIZE").getInt(null) / 8;
		} catch (IllegalArgumentException | IllegalAccessException | NoSuchFieldException | SecurityException e) {
			e.printStackTrace();
		}
		return gridMemorySize;
	}

//    static FloatBuffer getUnifiedFloatBuffer(Pointer pinnedMemory, CUdeviceptr devicePtr, long size) {
//	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
//	final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
//	byteBuffer.order(ByteOrder.nativeOrder());
//	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
//	return byteBuffer.asFloatBuffer();
//    }
//
//    public static CudaIntBuffer getUnifiedIntBuffer(Pointer pinnedMemory, CUdeviceptr devicePtr, int size) {
//	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
//	final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
//	byteBuffer.order(ByteOrder.nativeOrder());
//	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
//	return byteBuffer.asIntBuffer();
//    }
//
//    public static int[] getUnifiedIntArray(Pointer pinnedMemory, CUdeviceptr devicePtr, int size) {
//	int[] values = new int[size];
//	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
//	final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
//	byteBuffer.order(ByteOrder.nativeOrder());
//	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
//	return values;
//    }
//
//    public static ByteBuffer getUnifiedByteBuffer(Pointer pinnedMemory, CUdeviceptr devicePtr, int size) {
//	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
//	final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
//	byteBuffer.order(ByteOrder.nativeOrder());
//	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
//	return byteBuffer;
//    }

	/**
	 * Stop the executors and clean memory on registered CUObject
	 */
	public static void stop() {
		synchronized (cudaEngines) {
			cuCtxSynchronizeAll();
			for (Iterator<CudaEngine> iterator = cudaEngines.values().iterator(); iterator.hasNext();) {
				iterator.next().shutdown();
				iterator.remove();

			}
			// for (CudaEngine ce : cudaEngines.values()) {
			// ce.shutdown();
			// }
		}
	}

	/**
	 * Stop the executors and clean memory on registered CUObjectgr
	 */
	synchronized public static void freeMemory() {
		for (CudaEngine ce : cudaEngines.values()) {
			ce.freeCUObjectsMemory();
		}
	}

	/**
	 * Free memory from the currently registered CUObjects
	 */
	public void freeCUObjectsMemory() {
		exe.submit(new Runnable() {

			@Override
			public void run() {
				cuCtxSynchronize();
				for (CudaObject co : cudaObjects) {
					co.freeMemory();
				}
				JCudaDriver.cuCtxDestroy(context);
			}
		});
	}

	private synchronized void shutdown() {
		if (!exe.isShutdown()) {
			freeCUObjectsMemory();
		}
		exe.shutdown();
		try {
			System.err.println("cuda device " + Id + " freed ? " + exe.awaitTermination(10, TimeUnit.SECONDS));
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	public CUstream getNewCudaStream() {
		try {
			return exe.submit(() -> {
				final CUstream cudaStream = new CUstream();
				JCudaDriver.cuStreamCreate(cudaStream, CUstream_flags.CU_STREAM_NON_BLOCKING);
				return cudaStream;
			}).get();
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
		return null;
	}

	CUfunction getFunction(String kernelFunctionName, String kernelFileBaseName) {
		URL url = CudaEngine.class.getClassLoader().getResource("cuda/"+kernelFileBaseName+".ptx");
		try {
			url = new File(CudaPlatform.getCudasLibsPath().toFile(),kernelFileBaseName+".ptx").toURI().toURL();
		} catch (MalformedURLException e) {
			e.printStackTrace();
		}
		JCudaDriver.cuModuleLoad(myModule, url.getPath());
		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, myModule, kernelFunctionName);
		return function;
	}

	public int getMaxThreads() {
		return maxThreads;
	}

	public static synchronized void cuCtxSynchronizeAll() {
		for (CudaEngine ce : cudaEngines.values()) {
			ce.cuCtxSynchronize();
		}
	}

	public void cuCtxSynchronize() {
		try {
			exe.submit(JCudaDriver::cuCtxSynchronize).get();
		} catch (ExecutionException | InterruptedException e) {
			Thread.currentThread().interrupt();
		}
	}

	public Future<?> submit(Runnable runnable) {
		if (!exe.isShutdown()) {
			return exe.submit(runnable);
		}
		return null;
	}

	public void freeCudaMemory(Pointer p) {
		exe.submit(() -> cuMemFreeHost(p));
	}

	public void freeCudaMemory(CUdeviceptr p) {
		exe.submit(() -> cuMemFree(p));
	}

	/**
	 * Implements a little test that instantiates the CudaEngine and then cleans up
	 * 
	 * @param args
	 * @throws IOException
	 */
	public static void main(String[] args) throws IOException {
		CudaPlatform.init(Level.ALL);
		ProcessBuilder pb = new ProcessBuilder("nvcc", "--version");
		pb.inheritIO();
		pb.start();
		pb = new ProcessBuilder("env");
		pb.inheritIO();
		pb.start();
		CudaEngine cudaEngine = new CudaEngine(0);
		// KernelConfig kernelConfiguration =
		// cudaEngine.getDefaultKernelConfiguration(100, 100);
		// cudaEngine.getKernel("EVAPORATION",
		// "/turtlekit/cuda/kernels/Evaporation_2D.cu", kernelConfiguration);

	}


}
