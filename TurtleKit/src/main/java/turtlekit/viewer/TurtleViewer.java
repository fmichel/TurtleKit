package turtlekit.viewer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javafx.geometry.Rectangle2D;
import javafx.scene.paint.Color;
import javafx.stage.Screen;
import madkit.kernel.Probe;
import madkit.simulation.viewer.Viewer2D;
import turtlekit.agr.TKOrganization;
import turtlekit.kernel.Patch;
import turtlekit.kernel.TKEnvironment;
import turtlekit.kernel.TKGridModel;
import turtlekit.kernel.Turtle;
import turtlekit.kernel.TutleKit4;
import turtlekit.kernel.Utils;

/**
 * A viewer for the grid environment with turtles.
 * 
 */
public class TurtleViewer extends Viewer2D {

	private int cellSize = 7;
	private int[] onScreeenCellPosition;
//	private int[] onScreeenCellYPosition;

	@Override
	protected void onActivation() {
		addProbe(new Probe(getModelGroup(), TKOrganization.TURTLE_ROLE));
		super.onActivation();
		Rectangle2D screenBounds = Screen.getPrimary().getBounds();
		double screenW = screenBounds.getWidth();
		double screenH = screenBounds.getHeight();
		int width = getEnvironment().getWidth();
		int height = getEnvironment().getHeight();
		while ((width * cellSize > screenW - 300 || height * cellSize > screenH - 200) && cellSize > 1) {
			cellSize--;
		}
		getGUI().setCanvasSize(width * cellSize, height * cellSize);
		computeCellsOnScreenLocations();
	}

	@SuppressWarnings("unchecked")
	@Override
	public TKEnvironment<Patch> getEnvironment() {
		return (TKEnvironment<Patch>) super.getEnvironment();
	}

	/**
	 * 
	 */
	void computeCellsOnScreenLocations() {
		onScreeenCellPosition = new int[getWidth()];
		Arrays.parallelSetAll(onScreeenCellPosition, i -> i * cellSize);
	}

	/**
	 * 
	 */
	@Override
	public void render() {
		super.render();
		int index = 0;
		Patch[] grid = getPatchGrid();
		for (int j = getHeight() - 1; j >= 0; j--) {// cannot be parallel because of JavaFX
			for (int i = 0; i < getWidth(); i++) {
				Patch p = grid[index];
				Turtle<?> t = p.getTurtles().stream().filter(Turtle::isVisible).findAny().orElse(null);
				if (t == null) {
					paintPatch(p, onScreeenCellPosition[i], onScreeenCellPosition[j], index);
				} else {
					paintTurtle(t, onScreeenCellPosition[i], onScreeenCellPosition[j]);
				}
				index++;
			}
		}
	}

	public void paintTurtle(Turtle<?> t, int i, int j) {
		getGraphics().setFill(t.getColor());
		getGraphics().fillRect(i, j, cellSize, cellSize);
	}

	public void paintPatch(Patch p, int x, int y, int gridIndex) {
		getGraphics().setFill(p.getColor());
		getGraphics().fillRect(x, y, cellSize, cellSize);
	}

	public void clear() {
		getGraphics().setFill(Color.BLACK);
		getGraphics().fillRect(0, 0, getWidth() * cellSize, getHeight() * cellSize);// NOSONAR
	}

	public TKGridModel getGridModel() {
		return getEnvironment().getGridModel();
	}

	public Patch[] getPatchGrid() {
		return getGridModel().getPatchGrid();
	}

	/**
	 * @return
	 */
	public int getHeight() {
		return getEnvironment().getHeight();
	}

	/**
	 * @return
	 */
	public int getWidth() {
		return getEnvironment().getWidth();
	}

	/**
	 * @return the cellSize
	 */
	public int getCellSize() {
		return cellSize;
	}

	/**
	 * @param cellSize the cellSize to set
	 */
	public void setCellSize(int cellSize) {
		this.cellSize = cellSize;
	}

	protected static void executeThisViewer(String... args) {
		List<String> arguments = new ArrayList<>(List.of(args));
		String className = Utils.getClassFromMainStackTrace();
		arguments.addAll(List.of("-v", className));
		TutleKit4.main(arguments.toArray(new String[arguments.size()]));
	}

}