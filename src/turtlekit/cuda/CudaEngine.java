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

import static jcuda.driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import jcuda.CudaException;
import jcuda.Pointer;
import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;
import jcuda.utils.KernelLauncher;
import turtlekit.pheromone.Pheromone;

public class CudaEngine {

	private static final String PHEROMONES_CU = "pheromones";

	enum Kernel {
		DIFFUSION_TO_TMP, 
		DIFFUSION_UPDATE, 
		EVAPORATION, FIELD_MAX_DIR, 
		DIFFUSION_UPDATE_THEN_EVAPORATION, 
		DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIR
//		,FILL_NEIGHBORS
		,TEST
, DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIRV2
	}

	private static int availableDevicesNb = 0;

	private static ExecutorService initialization;

	final static Map<Integer,CudaEngine> cudaEngines = new HashMap<>();

	final HashMap<Kernel, CUfunction> kernels = new HashMap<Kernel, CUfunction>();

	private static final String ioTmpDir = System.getProperty("java.io.tmpdir");
	
	private static int NB_OF_DEVICE_TO_USE = 1;

	private static void extractAndLoadNativeLib(String nativeLibName, Path target){
//		System.err.println("loading "+nativeLibName);
		final Path path = Paths.get(target.toString(),nativeLibName);
		if (! path.toFile().exists()) {
			try (InputStream is = CudaEngine.class.getResourceAsStream("/lib/"	+ nativeLibName)) {//TODO TK property for lib dir
				Files.copy(is, path);
			} catch (IOException e) {
				e.printStackTrace();
			} catch (NullPointerException e) {//TODO find a way to do it instead of eclipse
				final Path eclipsePath = FileSystems.getDefault().getPath("lib",nativeLibName);
				try {
					Files.copy(eclipsePath, path);
				} catch (IOException e1) {
					// TODO Auto-generated catch block
					e1.printStackTrace();
				}
			}
		}
		System.load(path.toString());
//		System.load(nativeLibName);
	}
	
	
	private static void extractAndLoadNativeLibs() throws IOException{
		Path target = Paths.get(ioTmpDir, "/tklib");
		if(! target.toFile().exists()){
			Files.createDirectories(target);
		}
		final boolean windows = System.getProperty("os.name").equalsIgnoreCase("windows");
		String fileExtension = windows ? "dll" : "so";
		String prefix = windows ? "" : "lib";
		String libPattern = fileExtension.equals("dll") ? "-windows" : "-linux" + "-x86";
		if(System.getProperty("sun.arch.data.model").equals("64")){
			libPattern+="_64";
		}
		libPattern+="."+fileExtension;
//		System.err.println(libPattern);
		System.setProperty("java.library.path", target.toString());
//		System.err.println(System.getProperty("java.library.path"));
		extractAndLoadNativeLib(prefix+"JCudaDriver"+libPattern, target);
		extractAndLoadNativeLib(prefix+"JCudaRuntime"+libPattern, target);
		extractAndLoadNativeLib(prefix+"JCurand"+libPattern, target);
	}

	public static void main(String[] args) {
		init();
	}
	/**
	 * 
	 */
	public static boolean init() {
		synchronized (cudaEngines) {
			System.err.println("---------Initializing Cuda----------------");
			try {
				extractAndLoadNativeLibs();
				JCudaDriver.setExceptionsEnabled(true);
				JCudaDriver.cuInit(0);
				compileKernelsPtx();
				// Obtain the number of devices
				int deviceCountArray[] = { 0 };
				JCudaDriver.cuDeviceGetCount(deviceCountArray);
				availableDevicesNb = deviceCountArray[0];
				if (availableDevicesNb == 0)
					return false;
				availableDevicesNb = NB_OF_DEVICE_TO_USE;// TODO
				initialization = Executors.newCachedThreadPool();
				System.out.println("Found " + availableDevicesNb + " GPU devices");
				for (int i = 0/*-NB_OF_DEVICE_TO_USE*/; i < availableDevicesNb; i++) {
					final int index = i;
					Future<?> initJob = initialization.submit(new Runnable() {
						public void run() {
							System.err.println("Initializing device n°" + index);
							cudaEngines.put(index, new CudaEngine(index));
						}
					});
					initJob.get();
					initialization.shutdown();
				}
			} catch (InterruptedException | ExecutionException | IOException | CudaException | UnsatisfiedLinkError e) {
				e.printStackTrace();
				System.err.println("---------Cannot initialize Cuda !!! ----------------");
				return false;
			}
			Runtime.getRuntime().addShutdownHook(new Thread() {
				@Override
				public void run() {
					CudaEngine.stop();
				}
			});
			System.out.println("---------Cuda Initialized----------------");
			return true;
		}
	}

	public int cuDeviceGetCount() {
		return availableDevicesNb;
	}

	private static AtomicInteger cudaObjectID = new AtomicInteger(0);


	private ExecutorService exe;
	protected CUfunction f;
	private List<CudaObject> cudaObjects = new ArrayList<CudaObject>();

	private int maxThreads;

	private int Id = -1;

	private Map<String,CUdeviceptr> neigborsPtrs;

	protected CUcontext context;

	private CudaEngine(final int deviceId) {
		exe = Executors.newSingleThreadExecutor(); //mandatory: Only one cuda thread per context
		Id  = deviceId;
		try {
			exe.submit(new Runnable() {
				@Override
				public void run() {
					CUdevice device = new CUdevice();
					JCudaDriver.cuDeviceGet(device, deviceId);
					int array[] = { 0 };
					JCudaDriver.cuDeviceGetAttribute(array,
							CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
					maxThreads = (int) Math.sqrt(array[0]);
					context = new CUcontext();
//					JCudaDriver.cuCtxCreate(context, CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC, device);
					JCudaDriver.cuCtxCreate(context, 0, device);
					CUmodule m = new CUmodule();
					initModules(m);
					for (Kernel k : Kernel.values()) {
						initFunction(m, k);
					}
//					JCudaDriver.cuCtxSetCacheConfig(CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE);>
//					JCudaDriver.cuCtxSetSharedMemConfig(CUsharedconfig.CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE);
				}
			}).get();
		} catch (InterruptedException | ExecutionException e) {
			throw new RuntimeException(e.getMessage());
		}
		neigborsPtrs = new HashMap<>();
	}
	
	public static boolean isCudaAvailable() {
		return availableDevicesNb != 0;
	}

	public static CudaEngine getCudaEngine(CudaObject co) {
		synchronized (cudaEngines) {
			if (!isCudaAvailable())
				throw new CudaException("No cuda device found");
			try {
				initialization.awaitTermination(100, TimeUnit.SECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			Pheromone p = (Pheromone) co;
			final int pheroID = cudaObjectID.incrementAndGet();

			final CudaEngine ce = cudaEngines.get(pheroID % availableDevicesNb);
//			final CudaEngine ce = cudaEngines.get(1);
//			final CudaEngine ce = cudaEngines.get(0);
			
//			final CudaEngine ce;
//			if(p.getName().contains("PRE")){
//				ce = cudaEngines.get(0);
//			}
//			else{
//				ce = cudaEngines.get(1);
//			}
//			
			ce.cudaObjects.add(co);
			System.err.println(co+"ID "+pheroID+" getting cuda engine Id "+ce.Id);
			return ce;
		}
	}

	static FloatBuffer getUnifiedFloatBuffer(Pointer pinnedMemory,
			CUdeviceptr devicePtr, long size) {
		JCudaDriver.cuMemHostAlloc(pinnedMemory, size,
				JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
		final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
		byteBuffer.order(ByteOrder.nativeOrder());
		JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
		return byteBuffer.asFloatBuffer();
	}

	public static IntBuffer getUnifiedIntBuffer(Pointer pinnedMemory,
			CUdeviceptr devicePtr, 
			int size) {
		JCudaDriver.cuMemHostAlloc(pinnedMemory, size,
				JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
		final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
		byteBuffer.order(ByteOrder.nativeOrder());
		JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
		return byteBuffer.asIntBuffer();
	}

	public static int[] getUnifiedIntArray(Pointer pinnedMemory,
			CUdeviceptr devicePtr, int size) {
		int[] values = new int[size];
		JCudaDriver.cuMemHostAlloc(pinnedMemory, size,
				JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
		final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
		byteBuffer.order(ByteOrder.nativeOrder());
		JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
		
		return values;
	}

	public static ByteBuffer getUnifiedByteBuffer(Pointer pinnedMemory,
			CUdeviceptr devicePtr, int size) {
		JCudaDriver.cuMemHostAlloc(pinnedMemory, size,
				JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
		final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
		byteBuffer.order(ByteOrder.nativeOrder());
		JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
		return byteBuffer;
	}

	/**
	 * Stop the executors and clean memory on registered CUObject
	 */
	public static void stop() {
		synchronized (cudaEngines) {
			cuCtxSynchronizeAll();
			for (Iterator<CudaEngine> iterator = cudaEngines.values()
					.iterator(); iterator.hasNext();) {
				iterator.next().shutdown();
				iterator.remove();

			}
			//		for (CudaEngine ce : cudaEngines.values()) {
			//			ce.shutdown();
			//		}
		}
	}

	/**
	 * Stop the executors and clean memory on registered CUObject
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
		if (! exe.isShutdown()) {
			freeCUObjectsMemory();
		}
		exe.shutdown();
		try {
			System.err.println("cuda device "+Id+" freed ? " +exe.awaitTermination(10, TimeUnit.SECONDS));
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	static void initModules(CUmodule module) {
		JCudaDriver.cuModuleLoad(module, new File(ioTmpDir,PHEROMONES_CU+".ptx").getAbsolutePath());
	}

	private void initFunction(CUmodule module, Kernel name) {
		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module, name.name());
		kernels.put(name, function);
	}

	// void compileKernelsPtx()
	static void compileKernelsPtx() throws IOException {
		if(! new File(ioTmpDir,PHEROMONES_CU+".ptx").exists()){//TODO externalize
			try(InputStream is = CudaEngine.class.getResourceAsStream("/turtlekit/cuda/kernels/"+PHEROMONES_CU+".cu")){
				final Path path = Paths.get(ioTmpDir, PHEROMONES_CU+".cu");
				try {
					Files.copy(is, path);
				} catch (FileAlreadyExistsException e) {
				}
				System.err.println("--------------- Compiling ptx ----------------------");
				KernelLauncher.create(
						path.toString(),
						Kernel.DIFFUSION_TO_TMP.name(), false, "--use_fast_math","--prec-div=false");//,"--gpu-architecture=sm_20");
			} catch (IOException e) {
				throw e;
			}
		}
	}

	public int getMaxThreads() {
		return maxThreads;
	}

	public CUfunction getKernelFunction(Kernel f) {
		CUfunction function = kernels.get(f);
		if (function == null)
			throw new CudaException("No such function " + f);
		return function;
	}

//	public static void main(String[] args) {
//		CudaPheromone cu, cu2;
//		getCudaEngine(cu = new CudaPheromone(10, 10, 0.1f, 0f, "test"));
//		getCudaEngine(cu2 = new CudaPheromone(100, 100, 0.3f, 0.5f, "test2"));
//		cu.set(3, 3, 8);
////		cu.diffusion();
//		cu.evaporation();
//		System.err.println(cu.get(3, 3));
//		System.err.println(cu.get(0, 0));
//		System.err.println(cu.get(3, 2));
//		System.err.println("maxdir " + cu.getMaxDir(3, 2));
//		cu.diffusion();
//		System.err.println(cu.get(3, 3));
//		cu.diffusion();
//		System.err.println(cu.get(3, 3));
//		cu2.diffusion();
//		cu.freeMemory();
//		cu2.freeMemory();
//		CudaEngine.stop();
//	}

	public static synchronized void cuCtxSynchronizeAll() {
		for (CudaEngine ce : cudaEngines.values()) {
			ce.cuCtxSynchronize();
		}
		// List<Future<Void>> futures = new ArrayList<Future<Void>>();
		// for (CudaEngine ce : executors) {
		// futures.add(ce.exe.submit(new Callable<Void>() {
		// @Override
		// public Void call() throws Exception {
		// JCudaDriver.cuCtxSynchronize();
		// return null;
		// }
		//
		// }));
		// }
		// for (Future<Void> future : futures) {
		// try {
		// future.get();
		// } catch (InterruptedException | ExecutionException e) {
		// // TODO Auto-generated catch block
		// e.printStackTrace();
		// }
		// }
	}

	public void cuCtxSynchronize() {
		try {
			exe.submit(new Callable<Void>() {
				@Override
				public Void call() throws Exception {
					JCudaDriver.cuCtxSynchronize();
					return null;
				}
			}).get();
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
	}

	public Future<?> submit(Runnable runnable) {
		if (! exe.isShutdown()) {
			return exe.submit(runnable);
		}
		return null;
	}

	public CUdeviceptr getNeighborsPtr(String string) {
		return neigborsPtrs.get(string);
	}

	public void addNeighborsPtr(String string, CUdeviceptr neighborsPtr) {
		neigborsPtrs.put(string, neighborsPtr);
	}
	
}
