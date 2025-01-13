package turtlekit.kernel;

import static org.testng.Assert.assertEquals;

import java.util.List;

import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class TKGridModelTest {

	private TKGridModel<Patch> gridModel;

	@BeforeClass
	public void setUp() throws Exception {
		gridModel = new TKGridModel<Patch>(new TKEnvironment<>(), true, 400, 400, Patch.class);
	}

	@Test
	public void nbOfOfPatchesInRadiusTest() {
		int nb = gridModel.nbOfOfPatchesInRadius(0, 0, 1);
		assertEquals(nb, 9);
		nb = gridModel.nbOfOfPatchesInRadius(0, 0, 2);
		assertEquals(nb, 25);
		nb = gridModel.nbOfOfPatchesInRadius(0, 0, 3);
		assertEquals(nb, 49);
		gridModel = new TKGridModel<Patch>(new TKEnvironment<>(), true, 2, 2, Patch.class);
		nb = gridModel.nbOfOfPatchesInRadius(0, 0, 1);
		assertEquals(nb, 4);
		nb = gridModel.nbOfOfPatchesInRadius(0, 0, 2);
		assertEquals(nb, 4);
		gridModel = new TKGridModel<Patch>(new TKEnvironment<>(), true, 3, 3, Patch.class);
		nb = gridModel.nbOfOfPatchesInRadius(0, 0, 1);
		assertEquals(nb, 9);
		nb = gridModel.nbOfOfPatchesInRadius(0, 0, 2);
		assertEquals(nb, 9);
		nb = gridModel.nbOfOfPatchesInRadius(0, 0, 3);
		assertEquals(nb, 9);
	}
	
	@Test
	public void getNeighborsOfTest() {
		gridModel = new TKGridModel<Patch>(new TKEnvironment<>(), true, 400, 400, Patch.class);
		Patch p = gridModel.getPatch(0, 0);
		List<Patch> l = gridModel.getNeighborsOf(p, 2, false);
		for (int i = 0; i < 30; i++) {
			l = gridModel.getNeighborsOf(p, i, false);
		}
		System.err.println(l.size());
	}
}
