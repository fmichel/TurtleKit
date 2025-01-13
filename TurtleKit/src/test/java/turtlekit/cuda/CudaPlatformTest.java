package turtlekit.cuda;

import static org.testng.Assert.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Level;
import java.util.stream.Stream;

import org.testng.SkipException;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

public class CudaPlatformTest {

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
	public void compiledPtxFilesArePresent() {
		try {
			Stream<Path> l = Files.list(CudaPlatform.getCudasLibsPath());
			assertTrue(l.anyMatch(f -> f.toString().endsWith("ptx")));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Test
	public void isCudaAvailableTest() {
		CudaPlatform.init(Level.ALL);
	}
}
