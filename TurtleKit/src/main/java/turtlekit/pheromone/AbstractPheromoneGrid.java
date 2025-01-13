/**
 * 
 */
package turtlekit.pheromone;

import madkit.gui.SliderProperty;
import madkit.gui.UIProperty;

/**
 * @author Fabien Michel
 *
 */
public abstract class AbstractPheromoneGrid<T> extends DataGrid<T> implements Pheromone<T> {

	private T maximum;
	
	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = .01)
	@UIProperty(category = "Phero", displayName = "diffusion")
	private double diffusionCoef = 0.5;

	@SliderProperty(minValue = 0, maxValue = 1, scrollPrecision = .01)
	@UIProperty(category = "Phero", displayName = "evaporation")
	private double evaporationCoef = 0.5;

	/**
	 * @return the diffusionCoef
	 */
	public double getDiffusionCoef() {
		return diffusionCoef;
	}

	/**
	 * @param diffusionCoef the diffusionCoef to set
	 */
	public void setDiffusionCoef(double diffusionCoef) {
		this.diffusionCoef = diffusionCoef;
	}

	/**
	 * @return the evaporationCoef
	 */
	public double getEvaporationCoef() {
		return evaporationCoef;
	}

	/**
	 * @param evaporationCoef the evaporationCoef to set
	 */
	public void setEvaporationCoef(double evaporationCoef) {
		this.evaporationCoef = evaporationCoef;
	}

	/**
	 * @param name
	 * @param width
	 * @param height
	 */
	public AbstractPheromoneGrid(String name, int width, int height, final float evaporationCoeff, final float diffusionCoeff) {
		super(name, width, height);
		setEvaporationCoef(evaporationCoeff);
		setDiffusionCoef(diffusionCoeff);
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
	public final double getDiffusionCoefficient(){
		return diffusionCoef;
	}

	/**
	 * @return the evaporation coefficient as a float between 0 and 1, e.g. 0.33 for 33% 
	 */
	public final double getEvaporationCoefficient(){
		return evaporationCoef;
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
