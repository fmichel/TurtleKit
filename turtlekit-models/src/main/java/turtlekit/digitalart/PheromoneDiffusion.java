package turtlekit.digitalart;

import turtlekit.pheromone.PheroColorModel;
import turtlekit.pheromone.PheroColorModel.MainColor;
import turtlekit.pheromone.Pheromone;
import turtlekit.viewer.TKViewer;

public class PheromoneDiffusion extends TKViewer {

	@Override
	protected void onActivation() {
		Pheromone<Float> p = getEnvironment().getPheromone("one", .02f, 0.80f);
		p.incValue(getWidth() / 2 - 100, getHeight() / 2 - 100, 90000000000L);

		p = getEnvironment().getPheromone("two", .02f, 0.8f);
		p.incValue(getWidth() / 2 - 300, getHeight() / 2 + 30, 90000000000L);
		p.setColorModel(new PheroColorModel(0, 0, 0, MainColor.RED));

		p = getEnvironment().getPheromone("three", .02f, 0.8f);
		p.incValue(getWidth() / 2 - 30, getHeight() / 2 + 30, 90000000000L);
		p.setColorModel(new PheroColorModel(0, 0, 0, MainColor.GREEN));

		p = getEnvironment().getPheromone("four", .02f, 0.8f);
		p.incValue(getWidth() / 2 + 30, getHeight() / 2 - 100, 90000000000L);
		p.setColorModel(new PheroColorModel(0, 0, 0, MainColor.BLUE));

		p = getEnvironment().getPheromone("five", .02f, 0.8f);
		p.incValue(getWidth() / 2 + 240, getHeight() / 2 - 200, 90000000000L);
		p.setColorModel(new PheroColorModel(200, 200, 200, MainColor.RED));

		super.onActivation();
		selectAllPheros();
	}

	public static void main(String[] args) {
		executeThisViewer(
				"--width","500"
				,"--height","500"
//				,"--cuda"
				);
	}
}
