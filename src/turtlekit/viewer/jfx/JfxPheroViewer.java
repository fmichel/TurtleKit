package turtlekit.viewer.jfx;

import java.awt.Color;
import java.util.ConcurrentModificationException;
import java.util.Map;
import java.util.Random;

import javafx.scene.canvas.GraphicsContext;
import javafx.scene.paint.Paint;
import madkit.simulation.probe.SingleAgentProbe;
import turtlekit.agr.TKOrganization;
import turtlekit.kernel.Patch;
import turtlekit.kernel.TKEnvironment;
import turtlekit.kernel.TKScheduler;
import turtlekit.kernel.Turtle;
import turtlekit.pheromone.Pheromone;
import turtlekit.pheromone.PheromoneView;


public class JfxPheroViewer extends JfxDefaultViewer {

	private SingleAgentProbe<TKEnvironment, Map<String, Pheromone<Float>>> pheroProbe;
	private Pheromone<Float> selectedPheromone;
	private double max;
	private PheromoneView defaultView;
	private int redCanal;
	private int blueCanal;
	private int greenCanal;
	@Override
	protected void initProbes() {
		super.initProbes();
		pheroProbe = new SingleAgentProbe<>(
				getCommunity(), TKOrganization.MODEL_GROUP,
				TKOrganization.ENVIRONMENT_ROLE, "pheromones");
		addProbe(pheroProbe);
	}
	
	public Pheromone<Float> setSelectedPheromone(String name) {
		selectedPheromone = pheroProbe.getPropertyValue().get(name);
		if(selectedPheromone == null){
			for(Pheromone<Float> p : pheroProbe.getPropertyValue().values()){
				selectedPheromone = p;
				break;//TODO ugly
			}
		}
		if (selectedPheromone != null) {
			defaultView = new PheromoneView(selectedPheromone);
		}
		return selectedPheromone;
	}

	public void observe() {
	    
	}
	
	
	
    void observe2() {
//	System.err.println(Platform.isFxApplicationThread());
	setSelectedPheromone("ATT1");
		if (selectedPheromone != null) {
			max = Math.log10(selectedPheromone.getMaximum() + 1) / 256;
			if(max == 0)
				max = 1;
			redCanal = defaultView.getRed().getValue();
			greenCanal = defaultView.getGreen().getValue();
			blueCanal = defaultView.getBlue().getValue();
//			timer.setText(selectedPheromone.toString());
		}
	try {
	    int index = 0;
	    final Patch[] grid = getPatchGrid();
	    final int w = getWidth();
//	    clear();
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
    
    public JfxPheroViewer(GraphicsContext gc) {
	super(gc);
    }

    @Override
    protected void activate() {
	requestRole(getCommunity(), TKOrganization.ENGINE_GROUP,TKOrganization.VIEWER_ROLE);
	getLogger().info("I am alive");
	initProbes();
	SingleAgentProbe<TKScheduler,Double> p = new SingleAgentProbe<>(getCommunity(), TKOrganization.ENGINE_GROUP, TKOrganization.SCHEDULER_ROLE,"GVT");
	addProbe(p);
   }

	@Override
	public void paintPatch(final Patch p, final int x, final int y, final int index) {
		if (selectedPheromone != null) {
			final double value = selectedPheromone.get(index);
			if (value > 0) {
				int r = (int) (Math.log10(value + 1) / max);
				r += redCanal;
				if (r > 255)
					r = 255;
				gc.setFill(javafx.scene.paint.Color.rgb(r, greenCanal, blueCanal));
				gc.fillRect(x, y, cellSize, cellSize);
			}
			else {
//				gc.setFill(randomColor());
//				gc.setFill(javafx.scene.paint.Color.BLACK);
//				gc.fillRect(x, y, cellSize, cellSize);
			}
		}
	}
	
	public Pheromone<Float> getSelectedPheromone() {
		return selectedPheromone;
	}
	public void setSelectedPheromone(Pheromone<Float> selectedPheromone) {
		this.selectedPheromone = selectedPheromone;
	}
	
	    public Paint randomColor() {
	        Random random = new Random();
	        int r = random.nextInt(255);
	        int g = random.nextInt(255);
	        int b = random.nextInt(255);
	        return javafx.scene.paint.Color.rgb(r, g, b);
	    }


    public void clear() {
		gc.setFill(javafx.scene.paint.Color.BLACK);
	gc.fillRect(0, 0, getWidth()*cellSize, getHeight()*cellSize);
    }

    public void switchToJPaintColor(Color awtColor) {
	int r = awtColor.getRed();
	int g = awtColor.getGreen();
	int b = awtColor.getBlue();
	int a = awtColor.getAlpha();
	double opacity = a / 255.0;
	gc.setFill(javafx.scene.paint.Color.rgb(r, g, b, opacity));
    }
 

}
