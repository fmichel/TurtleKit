package turtlekit.pheromone;

import javafx.scene.paint.Color;

/**
 * @param <T>
 */
/**
 * @param <T>
 */
public interface Pheromone<T> {

	/**
	 * Apply the diffusion process. The algorithm is a two-step process: 1. Diffuse
	 * the values to a temporary grid 2. Update the values in the main grid Default
	 * implementation is to call the two methods sequentially.
	 * 
	 * <pre>
	 * Default implementation is as follows: {@code} if (getDiffusionCoefficient()
	 * != 0) { { diffuseValuesToTmpGridKernel(); diffusionUpdateKernel(); } {@code}
	 */
	public default void diffusion() {
		if (getDiffusionCoefficient() != 0) {
			diffuseValuesToTmpGridKernel();
			updateValuesFromTmpGridKernel();
		}
	}

	/**
	 * Apply the diffusion and evaporation process. If both the diffusion and
	 * evaporation coefficients are not zero, the two processes are
	 * {@link #diffuseValuesToTmpGridKernel()} and
	 * {@link #diffusionAndEvaporationUpdateKernel()} so that the number of
	 * computations is minimized. Otherwise, the two processes are called
	 * sequentially. Default implementation is as follows: {@code} if
	 * (getDiffusionCoefficient() != 0 && getEvaporationCoefficient() != 0) { {
	 * diffuseValuesToTmpGridKernel(); diffusionAndEvaporationUpdateKernel(); }
	 * {@code} else { diffusion(); evaporation(); }
	 */
	public default void diffusionAndEvaporation() {
		if (getDiffusionCoefficient() != 0 && getEvaporationCoefficient() != 0) {
			diffuseValuesToTmpGridKernel();
			diffusionAndEvaporationUpdateKernel();
		} else {
			diffusion();
			evaporation();
		}
	}

	public void updateMaxValue();

	/**
	 * Apply the evaporation process. Default implementation is as follows:
	 * {@code} if (getEvaporationCoefficient() != 0) { { evaporationKernel(); }
	 * {@code}
	 */
	public default void evaporation() {
		if (getEvaporationCoefficient() != 0) {
			evaporationKernel();
		}
	}

	/**
	 * Apply the evaporation process. It should be overridden by the implementing
	 * class to apply the evaporation which is about reducing the value of each cell
	 * by a certain percentage.
	 */
	public void evaporationKernel();

	/**
	 * Apply the diffusion update and evaporation processes at once. This method is
	 * called after the {@link #diffuseValuesToTmpGridKernel()} method in the
	 * {@link #diffusionAndEvaporation()} method.
	 */
	public void diffusionAndEvaporationUpdateKernel();

	/**
	 * Diffuse the values to a temporary grid. This method is called before the
	 * {@link #updateValuesFromTmpGridKernel()} method.
	 */
	public void diffuseValuesToTmpGridKernel();

	/**
	 * Update the values in the main grid from the temporary grid. This method is
	 * called after the {@link #diffuseValuesToTmpGridKernel()} method.
	 */
	public void updateValuesFromTmpGridKernel();

	/**
	 * @return the diffusion coefficient between 0 and 1, e.g. 0.33 for 33%
	 */
	public double getDiffusionCoefficient();

	/**
	 * Set the diffusion coefficient between 0 and 1, e.g. 0.33 for 33%
	 * 
	 * @param diffusionCoef the diffusionCoef to set
	 */
	public void setDiffusionCoefficient(double diffusionCoef);

	/**
	 * Return the evaporation coefficient
	 * 
	 * @return the evaporation coefficient between 0 and 1, e.g. 0.33 for 33%
	 */
	public double getEvaporationCoefficient();

	/**
	 * Set the evaporation coefficient between 0 and 1, e.g. 0.33 for 33%
	 * 
	 * @param evaporationCoef the evaporationCoef to set
	 */
	public void setEvaporationCoefficient(double evaporationCoef);

	/**
	 * Update the maximum value in the grid
	 */
//	public void updateMaxValue();

	/**
	 * Returns the direction with the maximum value
	 * 
	 * @param x x coordinate
	 * @param y y coordinate
	 * @return the direction with the maximum value
	 */
	public int getMaxDirection(int x, int y);

	/**
	 * Returns the direction with the minimum value
	 * 
	 * @param x x coordinate
	 * @param y y coordinate
	 * @return the direction with the minimum value
	 */
	public int getMinDirection(int x, int y);

	/**
	 * Returns the value at the specified coordinates
	 * 
	 * @param x x coordinate
	 * @param y y coordinate
	 * @return the value at the specified coordinates
	 */
	public T get(int x, int y);

	/**
	 * Returns the value at the specified index
	 * 
	 * @param get1dIndex the index
	 * @return the value at the specified index
	 */
	public abstract T get(int get1dIndex);

	/**
	 * Returns the current maximum value in the grid
	 * 
	 * @return the current maximum value in the grid
	 */
	public abstract T getMaxEncounteredValue();

	public abstract void setMaxEncounteredValue(T value);

	/**
	 * @return the width
	 */
	public int getWidth();

	/**
	 * @return the height
	 */
	public int getHeight();

	/**
	 * @return the name
	 */
	public String getName();

	public int get1DIndex(int x, int y);

	public abstract void set(int index, T value);

	public void set(int x, int y, T value);

	public void incValue(int x, int y, float quantity);

	public void incValue(int code, float attractQty);

	public default Color getValueColor(T value) {
		return getColorModel().getColor(this, value);
	}

	public PheroColorModel getColorModel();

	public void setColorModel(PheroColorModel pheroColorMode);

	public default float getValueIntensity(float value) {
		return value / (float) getMaxEncounteredValue();
	}

	public default double getLogMaxValue() {
		return Math.log10((float) getMaxEncounteredValue() + 1) / 256;
	}

	public default void applyDynamics() {
		diffusionAndEvaporation();
		updateMaxValue();
//		computeGradients();
	}

//	public void computeGradients();

}
