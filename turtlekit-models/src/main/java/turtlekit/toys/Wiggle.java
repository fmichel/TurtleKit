package turtlekit.toys;

import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import turtlekit.kernel.DefaultTurtle;
import turtlekit.viewer.TKViewer;
import turtlekit.viewer.jfx.TurtlePopulationChartDrawer;

public class Wiggle extends DefaultTurtle {

	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = 0.01)
	@UIProperty(category = "Environment")
	private
	static double dyingRate = 0.1;

	@Override
	protected void onActivation() {
		super.onActivation();
		changeNextBehavior("wiggle");
		playRole("wiggler");
	}

	@Override
	public void wiggle() {
		super.wiggle();
		if (getCurrentBehaviorCount() > 20) {
			createTurtleHere(new Wiggle());
			setCurrentBehaviorCount(0);
		}
		if (getPatch().getTurtles().size()>1 && prng().nextDouble() < getDyingRate()) {
			die();
			return;
		}
	}

	public static void main(String[] args) {
		executeThisTurtle(1
				,"-v",TurtlePopulationChartDrawer.class.getName()
				,"-v",TKViewer.class.getName()
//				,"--headless"
				);
	}

	/**
	 * @return the dyingRate
	 */
	public static double getDyingRate() {
		return dyingRate;
	}

	/**
	 * @param dyingRate the dyingRate to set
	 */
	public static void setDyingRate(double dyingRate) {
		Wiggle.dyingRate = dyingRate;
	}

}
