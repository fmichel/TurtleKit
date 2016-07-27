/**
 * 
 */
package turtlekit.pheromone;

/**
 * @author Fabien Michel
 *
 */
public abstract class AbstractPheromoneGrid<T> extends DataGrid<T> {

	private CoefficientBoundedRangeModel diffusionCoefficient, evaporationCoefficient;
	private T maximum;

	/**
	 * @param name
	 * @param width
	 * @param height
	 */
	public AbstractPheromoneGrid(String name, int width, int height, final float evaporationCoeff, final float diffusionCoeff) {
		super(name, width, height);
		diffusionCoefficient = new CoefficientBoundedRangeModel(diffusionCoeff);
		evaporationCoefficient = new CoefficientBoundedRangeModel(evaporationCoeff);
	}
	
	/**
	 * @return the maximum
	 */
	public T getMaximum() {
		return maximum;
	}

	/**
	 * @param maximum the maximum to set
	 */
	public void setMaximum(T maximum) {
		this.maximum = maximum;
	}

	/**
	 * @return the diffusion coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
	 */
	final public float getDiffusionCoefficient(){
		return diffusionCoefficient.getCoefficient();
	}

	/**
	 * @return the diffusion coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
	 */
	final public CoefficientBoundedRangeModel getDiffusionCoefficientModel(){
		return diffusionCoefficient;
	}

	/**
	 * @return the evaporation coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
	 */
	final public CoefficientBoundedRangeModel getEvaporationCoefficientModel(){
		return evaporationCoefficient;
	}
	
	/**
	 * @return the evaporation coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
	 */
	final public float getEvaporationCoefficient(){
		return evaporationCoefficient.getCoefficient();
	}
	
	protected abstract void diffuseValuesToTmpGridKernel();
	
	protected abstract void diffusionUpdateKernel();

	public void diffusion(){
		if (getDiffusionCoefficient() != 0) {
			diffuseValuesToTmpGridKernel();
			diffusionUpdateKernel();
		}
	}
	
	/**
	 * This is faster than calling them sequentially: 
	 * Only one GPU kernel is called.
	 * 
	 */
	public void diffusionAndEvaporation() {
		if(getDiffusionCoefficient() != 0 && getEvaporationCoefficient() != 0){
			diffuseValuesToTmpGridKernel();
			diffusionAndEvaporationUpdateKernel();
		}
		else{
			diffusion();
			evaporation();
		}
	}
	
	

	public abstract void diffusionAndEvaporationUpdateKernel();

	public void evaporation() {
		if (getEvaporationCoefficient() != 0) {
			evaporationKernel();
		}
	}

	public abstract void evaporationKernel();
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
	
public abstract int getMaxDirection(int xcor, int ycor) ;

	public abstract int getMinDirection(int i, int j);
}
