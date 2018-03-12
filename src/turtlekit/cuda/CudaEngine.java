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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

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
     * Native libs have to be loaded from the file system. This is the dir where native libs are extracted (in the
     * default io dir of the OS)
     */
    public static final String ioTmpDir = System.getProperty("java.io.tmpdir");

    private static int availableDevicesNb = 0;

    private static ExecutorService initialization;

    final static Map<Integer, CudaEngine> cudaEngines = new HashMap<>();

    private static int NB_OF_DEVICE_TO_USE = 1;

    private static Logger logger;

    /**
     * @param logLevel
     */
    public static boolean init(String logLevel) {
	logger = Logger.getLogger(CudaEngine.class.getSimpleName());
	logger.setLevel(Level.parse(logLevel));
	logger.setLevel(Level.ALL);
	synchronized (cudaEngines) {
	    logger.finer("---------Initializing Cuda----------------");
	    try {
		JCudaDriver.setExceptionsEnabled(true);
		JCudaDriver.cuInit(0);
		// Obtain the number of devices
		int deviceCountArray[] = { 0 };
		JCudaDriver.cuDeviceGetCount(deviceCountArray);
		availableDevicesNb = deviceCountArray[0];
		if (availableDevicesNb == 0)
		    return false;
		// availableDevicesNb = NB_OF_DEVICE_TO_USE;// TODO
		initialization = Executors.newCachedThreadPool();
		logger.finer("Found " + availableDevicesNb + " GPU devices");
		Future<?> initJob = initialization.submit(new Runnable() {

		    public void run() {
			for (int i = 0/*-NB_OF_DEVICE_TO_USE*/; i < availableDevicesNb; i++) {
			    final int index = i;
			    logger.finer("Initializing device n°" + index);
			    cudaEngines.put(index, new CudaEngine(index));
			}
		    }
		});
		initJob.get();
		initialization.shutdown();
	    }
	    catch(InterruptedException | ExecutionException | CudaException | UnsatisfiedLinkError e) {
		logger.finer("---------Cannot initialize Cuda !!! ----------------");
		e.printStackTrace();
		return false;
	    }
	    Runtime.getRuntime().addShutdownHook(new Thread() {

		@Override
		public void run() {
		    CudaEngine.stop();
		}
	    });
	    logger.fine("---------Cuda Initialized----------------");
	    return true;
	}
    }
    //
    // public Pointer getPointerToFloat(float f){
    // return Pointer.to(new float[]{f});
    // }

    public int cuDeviceGetCount() {
	return availableDevicesNb;
    }

    private static AtomicInteger cudaObjectID = new AtomicInteger(0);

    private static Map<CudaObject, CudaEngine> engineBinds = new HashMap<>();

    private ExecutorService exe;
    protected CUfunction f;
    private List<CudaObject> cudaObjects = new ArrayList<CudaObject>();

    private int maxThreads;

    private int Id = -1;

    protected CUcontext context;

    protected CUmodule myModule;

    private Map<String, CUfunction> functions = new HashMap<>();

    private CudaEngine(final int deviceId) {
	exe = Executors.newSingleThreadExecutor(new ThreadFactory() {

	    @Override
	    public Thread newThread(Runnable r) {
		Thread thread = new Thread(r);
		thread.setDaemon(true);
		return thread;
	    }
	}); // mandatory: Only one cuda thread per context
	Id = deviceId;
	try {
	    exe.submit(new Runnable() {

		@Override
		public void run() {
		    CUdevice device = new CUdevice();
		    JCudaDriver.cuDeviceGet(device, deviceId);
		    int array[] = { 0 };
		    JCudaDriver.cuDeviceGetAttribute(array, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
		    maxThreads = (int) Math.sqrt(array[0]);
		    // System.out.println(maxThreads);
		    // maxThreads = (int) 1024;
		    context = new CUcontext();
		    JCudaDriver.cuCtxCreate(context, 0, device);
		    myModule = new CUmodule();
		}
	    }).get();
	}
	catch(InterruptedException | ExecutionException e) {
	    throw new RuntimeException(e.getMessage());
	}
    }

    public static boolean isCudaAvailable() {
	return availableDevicesNb != 0;
    }

    @Override
    public String toString() {
	return "cudaEngine device #" + Id;
    }

    public static void addKernelSourceFile(String aPathInTheClassPath) {

    }

    public KernelConfiguration getDefaultKernelConfiguration(int dataSizeX, int dataSizeY) {
	int gridSizeX = (dataSizeX + maxThreads - 1) / maxThreads;
	int gridSizeY = (dataSizeY + maxThreads - 1) / maxThreads;
	return new KernelConfiguration(gridSizeX, gridSizeY, maxThreads, maxThreads, getNewCudaStream());
    }

    public CudaKernel getKernel(final String kernelFunctionName, final String cuSourceFilePath, final KernelConfiguration kc) {
	try {
	    return exe.submit(() -> {
		CUfunction function = functions.computeIfAbsent("" + kernelFunctionName + cuSourceFilePath, k -> updateCuSourceFile(kernelFunctionName, cuSourceFilePath));
		return new CudaKernel(function, CudaEngine.this, kernelFunctionName, cuSourceFilePath, kc);
	    }).get();
	}
	catch(InterruptedException | ExecutionException e) {
	    e.printStackTrace();
	}
	return null;
    }

    public <T> CUdeviceptr createDeviceDataGrid(int wdith, int height, Class<T> dataType) {
	CUdeviceptr tmpPtr = new CUdeviceptr();
	try {
	    submit(() -> cuMemAlloc(tmpPtr, getRequiredMemorySize(dataType, wdith, height))).get();
	}
	catch(InterruptedException | ExecutionException | IllegalArgumentException | SecurityException e) {
	    e.printStackTrace();
	}
	return tmpPtr;
    }

    public <T> Buffer getUnifiedBufferBetweenPointer(Pointer hostData, CUdeviceptr deviceData, Class<T> dataType, int wdith, int height) {
	try {
	    int size = getRequiredMemorySize(dataType, wdith, height);
	    final Buffer buffer = exe.submit(() -> {
		JCudaDriver.cuMemHostAlloc(hostData, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
		final ByteBuffer byteBuffer = hostData.getByteBuffer(0, size);
		byteBuffer.order(ByteOrder.nativeOrder());
		JCudaDriver.cuMemHostGetDevicePointer(deviceData, hostData, 0);
		return byteBuffer;
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
	    final Method method = buffer.getClass().getMethod("as" + simpleName + "Buffer");
	    method.setAccessible(true);
	    return (Buffer) method.invoke(buffer);
	}
	catch(InterruptedException | ExecutionException | IllegalArgumentException | IllegalAccessException | SecurityException | InvocationTargetException
		| NoSuchMethodException e) {
	    e.printStackTrace();
	}
	return null;
    }

    private <T> int getRequiredMemorySize(Class<T> dataType, int wdith, int height) {
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
	int floatGridMemorySize = 0;
	try {
	    floatGridMemorySize = wdith * height * dataType.getField("SIZE").getInt(null) / 8;
	}
	catch(IllegalArgumentException | IllegalAccessException | NoSuchFieldException | SecurityException e) {
	    // TODO Auto-generated catch block
	    e.printStackTrace();
	}
	return floatGridMemorySize;
    }

    public static CudaEngine getCudaEngine(CudaObject co) {
	synchronized (cudaEngines) {
	    if (!isCudaAvailable())
		throw new CudaException("No cuda device found");
	    try {
		initialization.awaitTermination(10, TimeUnit.SECONDS);
	    }
	    catch(InterruptedException e) {
		e.printStackTrace();
	    }
	    synchronized (engineBinds) {
		return engineBinds.computeIfAbsent(co, v -> {
		    final int pheroID = cudaObjectID.incrementAndGet();
		    final CudaEngine ce = cudaEngines.get(pheroID % availableDevicesNb);
		    // final CudaEngine ce = cudaEngines.get(0);
		    ce.cudaObjects.add(co);
		    logger.finer(co + "ID " + pheroID + " getting cuda engine Id " + ce.Id);
		    return ce;
		});
	    }
	}
    }

    static FloatBuffer getUnifiedFloatBuffer(Pointer pinnedMemory, CUdeviceptr devicePtr, long size) {
	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
	final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
	byteBuffer.order(ByteOrder.nativeOrder());
	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
	return byteBuffer.asFloatBuffer();
    }

    public static IntBuffer getUnifiedIntBuffer(Pointer pinnedMemory, CUdeviceptr devicePtr, int size) {
	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
	final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
	byteBuffer.order(ByteOrder.nativeOrder());
	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);
	return byteBuffer.asIntBuffer();
    }

    public static int[] getUnifiedIntArray(Pointer pinnedMemory, CUdeviceptr devicePtr, int size) {
	int[] values = new int[size];
	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
	final ByteBuffer byteBuffer = pinnedMemory.getByteBuffer(0, size);
	byteBuffer.order(ByteOrder.nativeOrder());
	JCudaDriver.cuMemHostGetDevicePointer(devicePtr, pinnedMemory, 0);

	return values;
    }

    public static ByteBuffer getUnifiedByteBuffer(Pointer pinnedMemory, CUdeviceptr devicePtr, int size) {
	JCudaDriver.cuMemHostAlloc(pinnedMemory, size, JCudaDriver.CU_MEMHOSTALLOC_DEVICEMAP);
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
	if (!exe.isShutdown()) {
	    freeCUObjectsMemory();
	}
	exe.shutdown();
	try {
	    System.err.println("cuda device " + Id + " freed ? " + exe.awaitTermination(10, TimeUnit.SECONDS));
	}
	catch(InterruptedException e) {
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
	}
	catch(InterruptedException | ExecutionException e) {
	    e.printStackTrace();
	}
	return null;
    }

    private CUfunction updateCuSourceFile(String kernelFunctionName, String dotCuSourceFilePath) {
	KernelLauncher.setCompilerPath("/usr/local/cuda-9.1/bin/");//FIXME
	try (final InputStream is = CudaEngine.class.getResourceAsStream(dotCuSourceFilePath)) {
	    final URL resource = CudaEngine.class.getResource(dotCuSourceFilePath);
	    if (resource == null || is == null)
		throw new FileNotFoundException(dotCuSourceFilePath + " not found on the class path");
	    String fileName = dotCuSourceFilePath.substring(dotCuSourceFilePath.lastIndexOf(File.separatorChar) + 1);
	    final Path path = Paths.get(CudaEngine.ioTmpDir, fileName);
	    final File file = path.toFile();
	    boolean rebuildNeeded = !file.exists();
	    try { // IDE mode
		File f = new File(resource.toURI());
		rebuildNeeded = rebuildNeeded || file.lastModified() < f.lastModified();
	    }
	    catch(IllegalArgumentException e) { // jar file mode
	    }
	    String cuFile = path.toString();
	    if (rebuildNeeded) {
		Files.copy(is, path, StandardCopyOption.REPLACE_EXISTING);
		System.err.println("--------------- Compiling ptx from " + cuFile);
	    }
	    KernelLauncher.create(cuFile, kernelFunctionName, rebuildNeeded, "--use_fast_math", "--prec-div=false");// ,"--gpu-architecture=sm_20");
	    JCudaDriver.cuModuleLoad(myModule, cuFile.substring(0, cuFile.lastIndexOf('.')) + ".ptx");
	    System.err.println("initializing kernel " + kernelFunctionName);

	    CUfunction function = new CUfunction();
	    JCudaDriver.cuModuleGetFunction(function, myModule, kernelFunctionName);
	    return function;
	}
	catch(URISyntaxException | IOException e) {
	    e.printStackTrace();
	}
	return null;
	// try(InputStream is = CudaEngine.class.getResourceAsStreéam(dotCuSourceFilePath)){

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
	    exe.submit(() -> JCudaDriver.cuCtxSynchronize()).get();
	}
	catch(ExecutionException | InterruptedException e) {
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
//	ProcessBuilder pb = new ProcessBuilder("/usr/local/cuda-9.1/bin/nvcc");
//	pb.inheritIO();
//	pb.start();
	init(Level.ALL.toString());
	CudaEngine cudaEngine = new CudaEngine(0);
	KernelConfiguration kernelConfiguration = cudaEngine.getDefaultKernelConfiguration(100, 100);
	cudaEngine.getKernel("EVAPORATION", "/turtlekit/cuda/kernels/Evaporation_2D.cu", kernelConfiguration);

    }

}
