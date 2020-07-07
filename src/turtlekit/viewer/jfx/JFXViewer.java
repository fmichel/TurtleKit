package turtlekit.viewer.jfx;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Toolkit;
import java.util.ConcurrentModificationException;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import madkit.gui.fx.FXUI;
import madkit.simulation.probe.SingleAgentProbe;
import madkit.simulation.viewer.FXViewer;
import turtlekit.agr.TKOrganization;
import turtlekit.kernel.Patch;
import turtlekit.kernel.TKEnvironment;
import turtlekit.kernel.TKGridModel;
import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit;

@FXUI
public class JFXViewer extends FXViewer {

    protected int cellSize = 5;
    private SingleAgentProbe<TKEnvironment, Patch[]> gridProbe;

    /**
     * @deprecated replaced by using directly {@link #getCurrentEnvironment()}
     */
    public SingleAgentProbe<TKEnvironment, Patch[]> getGridProbe() {
	return gridProbe;
    }

    private SingleAgentProbe<TKEnvironment, TKGridModel> gridModelProbe;
    SingleAgentProbe<TKEnvironment, TKGridModel> getGridModelProbe() {
	return gridModelProbe;
    }

    private SingleAgentProbe<TKEnvironment, Integer> widthProbe;
    private SingleAgentProbe<TKEnvironment, Integer> heightProbe;

    protected void initProbes() {
	gridModelProbe = new SingleAgentProbe<TKEnvironment, TKGridModel>(getCommunity(), TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "gridModel");
	addProbe(gridModelProbe);
	gridProbe = new SingleAgentProbe<TKEnvironment, Patch[]>(getCommunity(), TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "patchGrid");
	addProbe(gridProbe);
	widthProbe = new SingleAgentProbe<TKEnvironment, Integer>(getCommunity(), TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "width");
	addProbe(widthProbe);
	heightProbe = new SingleAgentProbe<TKEnvironment, Integer>(getCommunity(), TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "height");
	addProbe(heightProbe);
    }

    /**
     * 
     */
    @Override
    protected void render(GraphicsContext gc) {
	try {
	    int index = 0;
	    final Patch[] grid = getPatchGrid();
	    final int w = getWidth();
	    clear();
	    for (int j = getHeight() - 1; j >= 0; j--) {
		for (int i = 0; i < w; i++) {
		    final Patch p = grid[index];
		    if (p.isEmpty()) {
			paintPatch(p, i * cellSize, j * cellSize, index);
		    }
		    else {
			try {
			    for (final Turtle t : p.getTurtles()) {
				if (t.isVisible()) {
				    paintTurtle(t, i * cellSize, j * cellSize);
				    break;
				}
			    }
			    // paintTurtle(g, p.getTurtles().get(0), i * cellSize, j * cellSize);
			}
			catch(NullPointerException | IndexOutOfBoundsException e) {// for the asynchronous mode
			    e.printStackTrace();
			}
		    }
		    index++;
		}
	    }
	}
	catch(ConcurrentModificationException e) {// FIXME
	}
    }

    @Override
    protected void activate() {
	initProbes();
	initCanvas();
	requestRole(getCommunity(), TKOrganization.ENGINE_GROUP, TKOrganization.VIEWER_ROLE);
    }
    
    
    private void initCanvas() {
	final Dimension screenSize = Toolkit.getDefaultToolkit().getScreenSize();
	double screenW = screenSize.getWidth();
	double screenH = screenSize.getHeight();
	while((getWidth()*cellSize > screenW - 200 || getHeight()*cellSize > screenH-300) && cellSize > 1){
		cellSize--;
	}
	setCanvas(new Canvas(getWidth(), getHeight()));
    }

    public void paintTurtle(final Turtle t, final int i, final int j) {
	switchToJPaintColor(t.getColor());
	getGraphics().fillRect(i, j, cellSize, cellSize);
    }

    public void paintPatch(final Patch p, final int x, final int y, final int index) {
	switchToJPaintColor(p.getColor());
	getGraphics().fillRect(x, y, cellSize, cellSize);
    }

    public void clear() {
	getGraphics().setFill(javafx.scene.paint.Color.BLACK);
	getGraphics().fillRect(0, 0, getWidth() * cellSize, getHeight() * cellSize);
    }

    public void switchToJPaintColor(Color awtColor) {
	int r = awtColor.getRed();
	int g = awtColor.getGreen();
	int b = awtColor.getBlue();
	int a = awtColor.getAlpha();
	double opacity = a / 255.0;
	getGraphics().setFill(javafx.scene.paint.Color.rgb(r, g, b, opacity));
    }

    /**
     * shortcut for getMadkitProperty(TurtleKit.Option.community)
     * 
     * @return the community of the simulation this agent is in
     */
    public final String getCommunity() {
	return getMadkitProperty(TurtleKit.Option.community);
    }

    public Patch[] getPatchGrid() {
	// return gridProbe.getPropertyValue();
	return getGridModel().getPatchGrid();
    }

    public Patch getPatch(int x, int y) {
	// return gridProbe.getPropertyValue()[x + getWidth() * y];
	return getGridModel().getPatch(x, y);
    }

    public TKGridModel getGridModel() {
	return gridModelProbe.getPropertyValue();
    }

    /**
     * @return
     */
    public int getHeight() {
	return heightProbe.getPropertyValue();
    }

    /**
     * @return
     */
    public int getWidth() {
	return widthProbe.getPropertyValue();
    }

    public TKEnvironment getCurrentEnvironment() {
	return gridModelProbe.getProbedAgent();
    }

    /**
     * @return the cellSize
     */
    public int getCellSize() {
	return cellSize;
    }

    /**
     * @param cellSize
     *            the cellSize to set
     */
    public void setCellSize(int cellSize) {
	this.cellSize = cellSize;
    }

}