package turtlekit.cuda;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLDecoder;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.jar.JarFile;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

import jcuda.CudaException;
import jcuda.driver.JCudaDriver;
import madkit.kernel.AgentLogger;

public class CudaPlatform {

	private static Logger logger = Logger.getLogger("[CUDA] ");

	private static final Path CUDAS_LIBS_PATH = Paths.get(System.getProperty("java.io.tmpdir"), "tklib");

	private static int cudaDevicesCount = 0;

	public static boolean init(Level logLevel) {
		initLogging(logLevel);
		logger.fine("--------- Initializing Cuda ----------------");
		try {
			checkJavaLibraryPath();
			installCudaNativeLibs();
			initJCudaDriver();
			initCudaEngines();
			compilePtxFiles();
			logger.fine("--------- Cuda Initialized ----------------");
		} catch (CudaException e) {
			logger.log(Level.SEVERE,e,()->"--------- Cuda NOT Initialized ----------------");
		}
		return getAvailableDevicesCount() != 0;
	}

	private static void checkJavaLibraryPath() {
		String libPath = System.getProperty("java.library.path");
		if (!libPath.contains(CUDAS_LIBS_PATH.toString())) {
			logger.severe(() -> "java.library.path does not contain the Cuda native libs folder" + CUDAS_LIBS_PATH
					+ "/lib "
					+ " -> Please add the following to JVM arguments: -Djava.library.path=" + CUDAS_LIBS_PATH + "/lib");
			throw new CudaException("java.library.path does not contain " + CUDAS_LIBS_PATH);
		}
	}

	private static void initCudaEngines() {
		for (int i = 0; i < getAvailableDevicesCount(); i++) {
			logger.finer("Initializing device n°" + i);
			new CudaEngine(i);
		}
	}

	private static void initLogging(Level logLevel) {
		final ConsoleHandler ch = new ConsoleHandler();
		ch.setFormatter(AgentLogger.AGENT_FORMATTER);
		ch.setLevel(Level.ALL);
		logger.addHandler(ch);
		logger.setUseParentHandlers(false);
		logger.setLevel(logLevel);
	}

	private static void initJCudaDriver() {
		logger.finer(() -> "Initializing JCudaDriver");
		JCudaDriver.setExceptionsEnabled(true);
		try {
			JCudaDriver.cuInit(0);
		} catch (Throwable e) {
			e.printStackTrace();
		}
		logger.finer(() -> "JCudaDriver initialized");
		int deviceCountArray[] = { 0 };
		JCudaDriver.cuDeviceGetCount(deviceCountArray);
		cudaDevicesCount = deviceCountArray[0];
		logger.finer(() -> "Found " + cudaDevicesCount + " GPU devices");
	}

	private static void installCudaNativeLibs() {
		if (checkIfNativeLibsExist())
			return;
		logger.finer("Extracting native libs");
		List<File> l = getNativesLibJarFiles();
		extractNativeLibsFromJarFiles(l, CUDAS_LIBS_PATH);
	}

	private static boolean checkIfNativeLibsExist() {
		if (Files.exists(Paths.get(CUDAS_LIBS_PATH.toString(), "lib"))) {
			logger.finer(() -> "Native libs already exist in " + CUDAS_LIBS_PATH);
			return true;
		}
		return false;
	}

	static List<File> getNativesLibJarFiles() {
		List<File> list = getJarFilesOnModulePath();
		list.removeIf(f -> !f.getName().contains("-natives"));
		logger.finer(() -> "Found " + list.size() + " native libs " + list);
		return list;
	}

	public static List<File> getJarFilesOnModulePath() {
		List<File> jarFiles = new ArrayList<>();
		String modulePath = System.getProperty("jdk.module.path");
		Stream.of(modulePath.split(File.pathSeparator)).forEach(path -> {
			File file = new File(path);
			if (file.isDirectory()) {
				try (Stream<Path> paths = Files.walk(FileSystems.getDefault().getPath(path))) {
					paths.filter(Files::isRegularFile).filter(p -> p.toString().endsWith(".jar")).map(Path::toFile)
							.forEach(jarFiles::add);
				} catch (IOException e) {
					e.printStackTrace();
				}
			} else if (file.isFile() && file.getName().endsWith(".jar")) {
				jarFiles.add(file);
			}
		});
		return jarFiles;
	}

	static void extractNativeLibsFromJarFiles(List<File> libs, Path destPath) {
		logger.finer(() -> "Extracting native libs to " + destPath);
		try {
			Files.createDirectories(destPath);
			for (File file : libs) {
				try (ZipFile archive = new ZipFile(file)) {
					List<? extends ZipEntry> entries = archive.stream().sorted(Comparator.comparing(ZipEntry::getName))
							.toList();
					for (ZipEntry entry : entries) {
						Path entryDest = destPath.resolve(entry.getName());
						try {
							if (entry.isDirectory()) {
								Files.createDirectory(entryDest);
								continue;
							}
							logger.finest(() -> "Extracting native libs to " + entryDest);
							Files.copy(archive.getInputStream(entry), entryDest);
						} catch (FileAlreadyExistsException e) {
							logger.finest(() -> "File already exists: " + entryDest);
						}
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	static void compilePtxFiles() {
		try {
			URL url = CudaPlatform.class.getClassLoader().getResource("cuda");

			if (url != null) {
				logger.finer(() -> "Compiling cu files from " + url);
				if ("file".equals(url.getProtocol())) {
					// Handle the case where the resource is in a directory
					Path p = Path.of(url.toURI());
					try (Stream<Path> stream = Files.find(p, Integer.MAX_VALUE,
							(path, attrs) -> path.toString().endsWith(".cu"))) {
						stream.forEach(f -> compilePtx(f));
					}
				} else if ("jar".equals(url.getProtocol())) {
					// Handle the case where the resource is in a JAR file
					String jarPath = url.getPath().substring(5, url.getPath().indexOf("!")); // Extract the JAR file path
					try (JarFile jarFile = new JarFile(URLDecoder.decode(jarPath, "UTF-8"))) {
						// Specify the output directory where you want to extract the .cu files
						Path outputDir = getCudasLibsPath();
						jarFile.stream()
								.filter(entry -> entry.getName().endsWith(".cu"))
								.forEach(entry -> {
									try (InputStream is = jarFile.getInputStream(entry)) {
										String entryName = entry.getName();
										String fileName = entryName.substring(entryName.lastIndexOf('/'));
										Path outputPath = Paths.get(outputDir.toString(), fileName);
										Files.copy(is, outputPath, java.nio.file.StandardCopyOption.REPLACE_EXISTING);
										compilePtx(outputPath);
									} catch (IOException e) {
										e.printStackTrace();
									}
								});
					}
				}
			}
		} catch (IOException | URISyntaxException e) {
			e.printStackTrace();
		}
	}

	private static void compilePtx(Path f) {
		File source = f.toFile();
		final File target = buildTargetFile(source);
		if (source.lastModified() > target.lastModified()) {
			try {
				logger.finer(() -> "Compiling " + f + "\n\t-> " + target);
				buildCompileProcess(source, target).start().waitFor();
			} catch (IOException | InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

	private static File buildTargetFile(File source) {
		String fileName = source.getName();
		String baseName = fileName.substring(0, fileName.lastIndexOf('.'));
		return new File(CUDAS_LIBS_PATH.toFile(), File.separator + baseName + ".ptx");
	}

	private static ProcessBuilder buildCompileProcess(File source, File target) {
		return new ProcessBuilder("nvcc","-arch=compute_50"
//		, "-m" + System.getProperty("sun.arch.data.model"), "--use_fast_math", "--prec-div=false"
		, "-ptx", source.getAbsolutePath(), "-o", target.getAbsolutePath()).inheritIO();
	}

	/**
	 * @return the availableDevicesCount
	 */
	public static int getAvailableDevicesCount() {
		return cudaDevicesCount;
	}

	public static boolean isCudaAvailable() {
		return cudaDevicesCount != 0;
	}

	public static Path getCudasLibsPath() {
		return CUDAS_LIBS_PATH;
	}

}
