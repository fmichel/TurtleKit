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
import static jcuda.driver.JCudaDriver.cuMemAlloc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
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
import jcuda.driver.CUstream;
import jcuda.driver.CUstream_flags;
import jcuda.driver.JCudaDriver;
import jcuda.utils.KernelLauncher;

public class CudaEngine {

	/**
	 * Native libs have to be loaded from the file system. 
	 * This is the dir where native libs are extracted (in the default io dir of the OS)
	 */
	public static final String ioTmpDir = System.getProperty("java.io.tmpdir");

	private static final Path TK_NATIVE_LIBS_DIR_PATH = Paths.get(ioTmpDir, "/tklib");

	private static final String PHEROMONES_CU = "pheromones";

	
	
	enum Kernels {
		DIFFUSION_TO_TMP, 
		DIFFUSION_UPDATE, 
		EVAPORATION, FIELD_MAX_DIR, 
		DIFFUSION_UPDATE_THEN_EVAPORATION
//		,DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIR
//		,FILL_NEIGHBORS
		,TEST
		,DIFFUSION_UPDATE_THEN_EVAPORATION_THEN_FIELDMAXDIRV2
		,AVERAGE_DEPTH_1D_V2
//		,HEAT_DEPTH_1D_V2
//		,STATE_COMPUTATION
//		,NUMBER_NEIGHBORS_ALIVE
//		,HERE_COMPUTATION
	}
	

	private static int availableDevicesNb = 0;

	private static ExecutorService initialization;

	final static Map<Integer,CudaEngine> cudaEngines = new HashMap<>();

	final HashMap<Kernels, CUfunction> kernels = new HashMap<Kernels, CUfunction>();

	
	private static int NB_OF_DEVICE_TO_USE = 1;

	/**
	 * If not already done, gets from the classpath (jar file or /lib) a required native lib and load it
	 * @param pathToLib
	 * @param nativeLibName
	 */
	private static void extractAndLoadNativeLib(Path pathToLib, String nativeLibName){
		//		System.err.println("loading "+nativeLibName);
		final Path path = Paths.get(pathToLib.toString(),nativeLibName);
		if (! path.toFile().exists()) {
			try (InputStream is = CudaEngine.class.getResourceAsStream("/lib/"	+ nativeLibName)) {
				Files.copy(is, path);
				System.err.println(path.toAbsolutePath());
				System.loadLibrary(path.toAbsolutePath().toString());
			} catch (IOException e) {
				e.printStackTrace();
			} 
		}
//				System.load(nativeLibName);
	}

	
	private static void extractAndLoadNativeLibs() throws IOException{
		if(! TK_NATIVE_LIBS_DIR_PATH.toFile().exists()){
			Files.createDirectories(TK_NATIVE_LIBS_DIR_PATH);
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
		System.setProperty("java.library.path", TK_NATIVE_LIBS_DIR_PATH.toString());
		System.err.println(System.getProperty("java.library.path"));
		extractAndLoadNativeLib(TK_NATIVE_LIBS_DIR_PATH, prefix+"JCudaDriver"+libPattern);
		extractAndLoadNativeLib(TK_NATIVE_LIBS_DIR_PATH, prefix+"JCudaRuntime"+libPattern);
		extractAndLoadNativeLib(TK_NATIVE_LIBS_DIR_PATH, prefix+"JCurand"+libPattern);
	}

	/**
	 * Implements a little test that instantiates the CudaEngine and then cleans up
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		init();
		CudaEngine cudaEngine = new CudaEngine(0);
		KernelConfiguration kernelConfiguration = cudaEngine.getDefaultKernelConfiguration(100, 100);
		cudaEngine.getKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu", kernelConfiguration);

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
//				compileKernelsPtx();
				// Obtain the number of devices
				int deviceCountArray[] = { 0 };
				JCudaDriver.cuDeviceGetCount(deviceCountArray);
				availableDevicesNb = deviceCountArray[0];
				if (availableDevicesNb == 0)
					return false;
//				availableDevicesNb = NB_OF_DEVICE_TO_USE;// TODO
				initialization = Executors.newCachedThreadPool();
				System.out.println("Found " + availableDevicesNb + " GPU devices");
					Future<?> initJob = initialization.submit(new Runnable() {
						public void run() {
							for (int i = 0/*-NB_OF_DEVICE_TO_USE*/; i < availableDevicesNb; i++) {
								final int index = i;
							System.err.println("Initializing device n°" + index);
							cudaEngines.put(index, new CudaEngine(index));
						}}
					});
					initJob.get();
					initialization.shutdown();
			} catch (InterruptedException | ExecutionException | CudaException | UnsatisfiedLinkError | IOException e) {
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

	protected CUmodule myModule;

	private Map<String, CUfunction> functions = new HashMap<>();

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
//					System.out.println(maxThreads);
//					maxThreads = (int) 1024;
					context = new CUcontext();
//					JCudaDriver.cuCtxCreate(context, CUctx_flags.CU_CTX_SCHED_BLOCKING_SYNC, device);
					JCudaDriver.cuCtxCreate(context, 0, device);
//					JCudaDriver.cuCtxCreate(context, CUctx_flags.CU_CTX_MAP_HOST, device);
					myModule = new CUmodule();
//					initModules(myModule);
//					for (Kernels k : Kernels.values()) {
//						initFunction(myModule, k);
//					}
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
	
	public static void addKernelSourceFile(String aPathInTheClassPath){
		
	}
	
	public KernelConfiguration getDefaultKernelConfiguration(int dataSizeX, int dataSizeY){
		int gridSizeX = (dataSizeX + maxThreads - 1) / maxThreads;
		int gridSizeY = (dataSizeY + maxThreads - 1) / maxThreads;
		return new KernelConfiguration(gridSizeX, gridSizeY, maxThreads, maxThreads, getNewCudaStream());
	}
	
	public CudaKernel getKernel(final String kernelFunctionName, final String cuSourceFilePath, final KernelConfiguration kc){
		try {
			return exe.submit(new Callable<CudaKernel>() {
				public CudaKernel call() {
					CUfunction function = functions.computeIfAbsent(""+kernelFunctionName+cuSourceFilePath, k -> updateCuSourceFile(kernelFunctionName,cuSourceFilePath));
					return new CudaKernel(function, CudaEngine.this, kernelFunctionName, cuSourceFilePath, kc);
				}
			}).get();
		} catch (InterruptedException | ExecutionException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	

	
	public <T> CUdeviceptr createDeviceDataGrid(int wdith, int height, Class<T> dataType){
		CUdeviceptr tmpPtr = new CUdeviceptr();
		try {
			final int floatGridMemorySize = wdith * height * dataType.getField("SIZE").getInt(null) / 8;
			submit(new Runnable() {
				@Override
				public void run() {
					cuMemAlloc(tmpPtr, floatGridMemorySize);
				}
			}).get();
		} catch (InterruptedException | ExecutionException | IllegalArgumentException | IllegalAccessException | NoSuchFieldException | SecurityException e) {
			e.printStackTrace();
		}
		return tmpPtr;
	}

	public <T> Buffer getUnifiedBufferBetweenPointer(Pointer hostData, CUdeviceptr deviceData, Class<T> dataType, int wdith, int height){
		try {
			int size = getRequiredMemorySize(dataType, wdith, height);
			final Buffer buffer = exe.submit(new Callable<Buffer>() {
				@Override
				public Buffer call() {
					JCudaDriver.cuMemHostAlloc(hostData, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
					final ByteBuffer byteBuffer = hostData.getByteBuffer(0, size);
					byteBuffer.order(ByteOrder.nativeOrder());
					JCudaDriver.cuMemHostGetDevicePointer(deviceData, hostData, 0);
					return byteBuffer;
				}
			}).get();
			String simpleName = dataType.getSimpleName();
			switch (simpleName) {
			case "Integer":
				simpleName = "Int";
				break;
			case "Character":
				simpleName = "Char";
				break;
			default:
				break;
			}
			final Method method = buffer.getClass().getMethod("as"+simpleName+"Buffer");
			method.setAccessible(true);
			return (Buffer) method.invoke(buffer);
		} catch (InterruptedException | ExecutionException | IllegalArgumentException | IllegalAccessException | NoSuchFieldException | SecurityException | InvocationTargetException | NoSuchMethodException e) {
			e.printStackTrace();
		}
		return null;
	}


	private <T> int getRequiredMemorySize(Class<T> dataType, int wdith, int height)
			throws IllegalAccessException, NoSuchFieldException {
		String simpleName = dataType.getSimpleName();
		switch (simpleName) {
		case "int":
			
			break;

		default:
			break;
		}
		final int floatGridMemorySize = wdith * height * dataType.getField("SIZE").getInt(null) / 8;
		return floatGridMemorySize;
	}

	public static CudaEngine getCudaEngine(CudaObject co) {
		synchronized (cudaEngines) {
			if (!isCudaAvailable())
				throw new CudaException("No cuda device found");
			try {
				initialization.awaitTermination(10, TimeUnit.SECONDS);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			
//			Pheromone p = (Pheromone) co;
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
		JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
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

	private void initFunction(CUmodule module, Kernels name) {
		CUfunction function = new CUfunction();
		JCudaDriver.cuModuleGetFunction(function, module, name.name());
		System.err.println("initializing kernel "+name);
		kernels.put(name, function);
	}
	

	// void compileKernelsPtx()
	static void compileKernelsPtx() throws IOException {
//		KernelLauncher.setCompilerPath("/usr/local/cuda-7.5/bin/");
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
						Kernels.DIFFUSION_TO_TMP.name(), false, "--use_fast_math","--prec-div=false");//,"--gpu-architecture=sm_20");
			} catch (IOException e) {
				throw e;
			}
		}
	}
	
	public CUstream getNewCudaStream(){
		try {
			return exe.submit(() -> {
				final CUstream cudaStream = new CUstream();
				JCudaDriver.cuStreamCreate(cudaStream, CUstream_flags.CU_STREAM_NON_BLOCKING);
				return cudaStream;
			}).get();
		} catch (InterruptedException | ExecutionException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	private CUfunction updateCuSourceFile(String kernelFunctionName, String dotCuSourceFilePath) {
//		KernelLauncher.setCompilerPath("/usr/local/cuda-7.0/bin/");//FIXME
		try {
			final URL resource = CudaEngine.class.getResource(dotCuSourceFilePath);
			if(resource == null)
				throw new FileNotFoundException(dotCuSourceFilePath+" not found on the class path");
			File f = new File(resource.toURI());
			final Path path = Paths.get(CudaEngine.ioTmpDir, f.getName());
			final File file = path.toFile();
			final boolean rebuildNeeded = ! file.exists() || file.lastModified() < f.lastModified();
			String cuFile = path.toString();
			if(rebuildNeeded){
				Files.copy(f.toPath(), path, StandardCopyOption.REPLACE_EXISTING);
				System.err.println("--------------- Compiling ptx from "+cuFile);
			}
			KernelLauncher.create(
					cuFile,
					kernelFunctionName, rebuildNeeded, "--use_fast_math","--prec-div=false");//,"--gpu-architecture=sm_20");
			JCudaDriver.cuModuleLoad(myModule, cuFile.substring(0, cuFile.lastIndexOf('.')) + ".ptx");
			System.err.println("initializing kernel "+ kernelFunctionName);
			
			CUfunction function = new CUfunction();
			JCudaDriver.cuModuleGetFunction(function, myModule, kernelFunctionName);
			return function;
		} catch (URISyntaxException | IOException e) {
			e.printStackTrace();
		}
		return null;
//		try(InputStream is = CudaEngine.class.getResourceAsStreéam(dotCuSourceFilePath)){
		
	}


	public int getMaxThreads() {
		return maxThreads;
	}

	public CUfunction getKernelFunction(Kernels f) {
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
		} catch (ExecutionException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
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
