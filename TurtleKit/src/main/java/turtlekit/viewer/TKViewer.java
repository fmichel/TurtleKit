package turtlekit.viewer;

import java.util.ArrayList;
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

public class TKViewer extends Viewer2D {

	private int cellSize = 7;
	private int[] onScreeenCellXPosition;
	private int[] onScreeenCellYPosition;

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
		computeCellsOnScreenLocations(cellSize);
	}

	@SuppressWarnings("unchecked")
	@Override
	public TKEnvironment<Patch> getEnvironment() {
		return (TKEnvironment<Patch>) super.getEnvironment();
	}

	/**
	 * 
	 */
	void computeCellsOnScreenLocations(int cellSize) {
		this.cellSize = cellSize;
		onScreeenCellXPosition = new int[getWidth()];
		onScreeenCellYPosition = new int[getHeight()];
		for (int i = 0; i < onScreeenCellXPosition.length; i++) {
			onScreeenCellXPosition[i] = i * cellSize;
		}
		for (int i = 0; i < onScreeenCellYPosition.length; i++) {
			onScreeenCellYPosition[i] = i * cellSize;
		}
	}

	/**
	 * 
	 */
	@Override
	public void render() {
		int index = 0;
		Patch[] grid = getPatchGrid();
		int w = getWidth();
		clear();
		for (int j = getHeight() - 1; j >= 0; j--) {
			for (int i = 0; i < w; i++) {
				Patch p = grid[index];
				Turtle<?> t = null;
				List<Turtle<?>> turtles = new ArrayList(p.getTurtles());
				for (Turtle<?> tmp : turtles) {
					if (tmp != null && tmp.isVisible()) {
						t = tmp;
						break;
					}
				}
				if (t == null) {
					paintPatch(p, onScreeenCellXPosition[i], onScreeenCellYPosition[j], index);
				} else {
					paintTurtle(t, onScreeenCellXPosition[i], onScreeenCellYPosition[j]);
				}
				index++;
			}
		}
	}

	public void paintTurtle(final Turtle<?> t, final int i, final int j) {
		getGraphics().setFill(t.getColor());
		getGraphics().fillRect(i, j, cellSize, cellSize);
	}

	public void paintPatch(final Patch p, final int x, final int y, int index) {
		Color color = p.getColor();
		if (color != Color.BLACK) {
			getGraphics().setFill(color);
			getGraphics().fillRect(x, y, cellSize, cellSize);
		}
	}

	public void clear() {
		getGraphics().setFill(Color.BLACK);
		getGraphics().fillRect(0, 0, getWidth() * cellSize, getHeight() * cellSize);
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
		final List<String> arguments = new ArrayList<>(List.of(args));
		String className = Utils.getClassFromMainStackTrace();
		arguments.addAll(List.of("-v", className));
		TutleKit4.main(arguments.toArray(new String[arguments.size()]));
	}

}