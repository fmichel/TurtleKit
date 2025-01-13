package turtlekit.viewer.jfx;

import java.util.List;
import java.util.Map;
import java.util.Random;

import javafx.scene.paint.Paint;
import madkit.gui.FXExecutor;
import turtlekit.kernel.Patch;
import turtlekit.pheromone.PheroView;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.TKViewer;

public class FXPheroViewer extends TKViewer {

	private Pheromone<Float> selectedPheromone;
	private double max;
	private PheroView defaultView;
	private int redCanal;
	private int blueCanal;
	private int greenCanal;
	private List<String> pheromones;

	@Override
	protected void onActivation() {
		super.onActivation();
		FXExecutor.runAndWait(() -> getGUI().getMainPane()
				.setLeft(defaultView = new PheroView(this, getEnvironment().getPheromonesMap())));
		setSelectedPheromone("test");
	}

	public Pheromone<Float> setSelectedPheromone(String name) {
		Map<String, Pheromone<Float>> pheros = getEnvironment().getPheromonesMap();
		if (name != null) {
			selectedPheromone = pheros.get(name);
		} else {
			selectedPheromone = pheros.values().stream().findAny().orElse(null);
		}
		return selectedPheromone;
	}

	public void render() {
		if (selectedPheromone != null) {
			max = Math.log10(selectedPheromone.getMaximum() + 1) / 256;
			if (max == 0)
				max = 1;
			redCanal = 100;
			greenCanal = 100;
			blueCanal = 100;
		}
		super.render();
	}

	@Override
	public void paintPatch(final Patch p, final int x, final int y, int index) {
		if (selectedPheromone != null) {
			final double value = selectedPheromone.get(index);
			if (value > 0.1) {
				int r = (int) (Math.log10(value + 1) / max);
				r += redCanal;
				if (r > 255)
					r = 255;
				getGraphics().setFill(javafx.scene.paint.Color.rgb(r, greenCanal, blueCanal));
				getGraphics().fillRect(x, y, getCellSize(), getCellSize());
			} else {
				// gc.setFill(randomColor());
				// gc.setFill(javafx.scene.paint.Color.BLACK);
				// gc.fillRect(x, y, cellSize, cellSize);
			}
		}
	}

	public Pheromone<Float> getSelectedPheromone() {
		return selectedPheromone;
	}

	public Paint randomColor() {
		Random random = new Random();
		int r = random.nextInt(255);
		int g = random.nextInt(255);
		int b = random.nextInt(255);
		return javafx.scene.paint.Color.rgb(r, g, b);
	}

	/**
	 * @return the pheromones
	 */
	public List<String> getPheros() {
		return pheromones;
	}

	/**
	 * @param pheros the pheromones to set
	 */
	public void setPheros(List<String> pheros) {
		this.pheromones = pheros;
	}

}
