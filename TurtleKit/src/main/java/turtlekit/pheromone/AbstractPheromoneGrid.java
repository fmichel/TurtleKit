/**
 * 
 */
package turtlekit.pheromone;

import java.util.random.RandomGenerator;

import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;
import turtlekit.kernel.TKEnvironment;

/**
 * @author Fabien Michel
 *
 */
public abstract class AbstractPheromoneGrid<T> extends DataGrid<T>
		implements Pheromone<T> {

	private T maxEncounteredValue;
	private double logMaxValue = 0;
	private final RandomGenerator prng;
	
	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = .01)
	@UIProperty(category = "Phero", displayName = "diffusion")
	private double diffusionCoefficient = 0.5;

	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = .01)
	@UIProperty(category = "Phero", displayName = "evaporation")
	private double evaporationCoefficient = 0.5;

	private PheroColorModel colorModel;

	protected final int[] neighborsIndexes;
	/**
	 * @param name
	 * @param environment
	 * @param evaporationCoeff
	 * @param diffusionCoeff
	 */
	protected AbstractPheromoneGrid(String name, TKEnvironment<?> environment, float evaporationCoeff, float diffusionCoeff) {
		super(name, environment.getWidth(), environment.getHeight());
		colorModel = new PheroColorModel();
		setEvaporationCoefficient(evaporationCoeff);
		setDiffusionCoefficient(diffusionCoeff);
		prng = environment.prng();
		neighborsIndexes = environment.getNeighborsIndexes();
	}

	@Override
	public PheroColorModel getColorModel() {
		return colorModel;
	}

	public void setColorModel(PheroColorModel colorModel) {
		this.colorModel = colorModel;
	}

	/**
	 * @param diffusionCoef the diffusionCoef to set
	 */
	public void setDiffusionCoefficient(double diffusionCoef) {
		this.diffusionCoefficient = diffusionCoef;
	}

	/**
	 * @return the evaporationCoef
	 */
	public double getEvaporationCoefficient() {
		return evaporationCoefficient;
	}

	/**
	 * @param evaporationCoef the evaporationCoef to setAbstractPheromoneGrid
	 */
	public void setEvaporationCoefficient(double evaporationCoef) {
		this.evaporationCoefficient = evaporationCoef;
	}

	public T getMaxEncounteredValue() {
		return maxEncounteredValue;
	}

	/**
	 * @param value the maximum to set
	 */
	public void setMaxEncounteredValue(T value) {
		this.maxEncounteredValue = value;
	}

	/**
	 * @return the diffusion coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
	 */
	public double getDiffusionCoefficient() {
		return diffusionCoefficient;
	}

	/**
	 * helper
	 * 
	 * @param x
	 * @param width
	 * @return
	 */
	public int normeValue(int x, int width) {
		if (x < 0) // -1
			return width - 1;
		if (x == width)
			return 0;
		return x;
	}

	public double getLogMaxValue() {
		return logMaxValue;
	}

	public void setLogMaxValue(double logMaxValue) {
		this.logMaxValue = logMaxValue;
	}

	public RandomGenerator prng() {
        return prng;
    }

		public int[] getNeighborsIndexes() {
			return neighborsIndexes;
		}
	}