package turtlekit.pheromone;

import madkit.gui.UIProperty;

public class TestObjectProperty {
	
    @UIProperty(category="Palm",displayName="Male palm ratio")
    private float malePalmRatio = 0.05f;//percentage > 0 && < 1

	/**
	 * @return the malePalmRatio
	 */
	public float getMalePalmRatio() {
		return malePalmRatio;
	}

	/**
	 * @param malePalmRatio the malePalmRatio to set
	 */
	public void setMalePalmRatio(float malePalmRatio) {
		System.err.println(malePalmRatio);
		this.malePalmRatio = malePalmRatio;
	}

    
 
}
