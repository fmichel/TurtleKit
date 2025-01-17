package turtlekit.ants;

import static javafx.scene.paint.Color.BLACK;
import static javafx.scene.paint.Color.YELLOW;

import java.util.Random;

import turtlekit.kernel.Patch;
import turtlekit.viewer.TKViewer;

public class AntViewer extends TKViewer {
	/**
	 * Just to do some initialization work
	 */
	@Override
	protected void onActivation() {
		Random random = new Random(1);
		super.onActivation();
//		setSelectedPheromone("presence");
//		setSynchronousPainting(false);// fastest display mode
		double densityRate = 0.01;
		for (Patch patch : getPatchGrid()) {
			if (random.nextDouble() < densityRate) {
				patch.setColor(YELLOW);
				patch.dropObject("food", Integer.valueOf(random.nextInt(1000)));
			} else
				patch.setColor(BLACK);
		}
	}

	@Override
	public void paintPatch(Patch p, int x, int y, int index) {
		if(p.getColor() == YELLOW){
			getGraphics().setFill(YELLOW);
			getGraphics().fillRect(x, y, getCellSize(), getCellSize());
		}
		else
			super.paintPatch(p, x, y, index);
	}

}
