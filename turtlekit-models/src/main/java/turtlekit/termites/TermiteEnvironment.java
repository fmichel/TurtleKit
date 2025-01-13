package turtlekit.termites;

import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import turtlekit.kernel.Patch;
import turtlekit.kernel.TKEnvironment;

public class TermiteEnvironment extends TKEnvironment<Patch> {

	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = 0.01)
	@UIProperty(category = "Environment")
	public static double chipsDensity = 0.5;

	@Override
	protected void onActivation() {
		super.onActivation();
		askPatches(null);
	}
}
