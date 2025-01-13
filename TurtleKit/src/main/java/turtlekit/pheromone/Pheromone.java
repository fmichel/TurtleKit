package turtlekit.pheromone;

public interface Pheromone<T> {

	/**
	 * @return the diffusion coefficient as a float between 0 and 1, e.g. 0.33 for
	 *         33%
	 */
	public double getDiffusionCoef();

	/**
	 * @param diffusionCoef the diffusionCoef to set
	 */
	public void setDiffusionCoef(double diffusionCoef);

	/**
	 * @return the evaporation coefficient as a float between 0 and 1, e.g. 0.33 for
	 *         33%
	 */
	public double getEvaporationCoef();

	/**
	 * @param evaporationCoef the evaporationCoef to set
	 */
	public void setEvaporationCoef(double evaporationCoef);

	public int getMaxDirection(int xcor, int ycor);

	public int getMinDirection(int i, int j);

	public void evaporation();

	public void diffusion();

	public void diffusionAndEvaporation();

	public T get(int x, int y);

	public abstract T get(int get1dIndex);

	public abstract T getMaximum();

	public abstract void setMaximum(T maximum);

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
	

}
