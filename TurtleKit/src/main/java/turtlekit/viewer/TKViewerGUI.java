package turtlekit.viewer;

import javafx.geometry.Rectangle2D;
import javafx.stage.Screen;
import madkit.simulation.viewer.CanvasDrawerGUI;

public class TKViewerGUI extends CanvasDrawerGUI {

	public TKViewerGUI(TKViewer viewer) {
		super(viewer);
		Rectangle2D screenBounds = Screen.getPrimary().getBounds();
		double screenW = screenBounds.getWidth();
		double screenH = screenBounds.getHeight();
		int width = viewer.getEnvironment().getWidth();
		int height = viewer.getEnvironment().getHeight();
		int cellSize = 6;
		while ((width * cellSize > screenW - 200 || height * cellSize > screenH - 300) && cellSize > 1) {
			cellSize--;
		}
		setCanvasSize(width * cellSize, height * cellSize);
		viewer.computeCellsOnScreenLocations(cellSize);
	}

}
