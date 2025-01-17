package turtlekit.viewer;

import java.util.ArrayList;
import java.util.List;

import javafx.collections.ObservableList;
import madkit.gui.FXExecutor;
import turtlekit.kernel.Patch;
import turtlekit.pheromone.Pheromone;
import turtlekit.pheromone.PherosViewNode;

/**
 * TKViewer is the default viewer for TurtleKit simulations. It extends
 * TurtleViewer and adds a view node for pheromones if there are any in the
 * environment.
 */
public class TKViewer extends TurtleViewer {

	/**
	 * need to keep a reference to the selected pheromones to avoid recomputing them
	 * during the rendering
	 */
	private List<Pheromone<Float>> selectedPherosCache = new ArrayList<>();
	private PherosViewNode pherosViewNode;

	@Override
	protected void onActivation() {
		super.onActivation();
		if (!getEnvironment().getPheromones().isEmpty()) {
			FXExecutor.runAndWait(() -> getGUI().getMainPane().setLeft(pherosViewNode = new PherosViewNode(this)));
		}
	}

	public void render() {
		updateSelectedPherosCache();
		super.render();
	}

	/**
	 * Update selected pheromones from the view node.
	 */
	void updateSelectedPherosCache() {
		if (pherosViewNode != null) {
			ObservableList<String> l = pherosViewNode.getSelectedPheromones();
			var pheromonesMap = getEnvironment().getPheromonesMap();
			selectedPherosCache = pheromonesMap.keySet().stream().filter(l::contains).map(pheromonesMap::get).toList();
		}
	}

	/**
	 * Get the pheromone with the maximum intensity on the cell at the given index.
	 * If no pheromone is selected, return null.
	 * 
	 * @param index the index of the current cell
	 * @return the pheromone with the maximum intensity on the cell at the given
	 *         index
	 */
	public Pheromone<?> getPheroWithMaxIntensity(int index) {
		if (selectedPherosCache.isEmpty())
			return null;
		Pheromone<?> maxPhero = selectedPherosCache.get(0);
		if (selectedPherosCache.size() > 1) {
			float maxValue = (float) maxPhero.get(index);
			float maxIntensity = maxPhero.getValueIntensity(maxValue);
			for (Pheromone<?> phero : selectedPherosCache.subList(1, selectedPherosCache.size())) {
				float value = (float) phero.get(index);
//				if (value > 0.0001) {
				float intensity = phero.getValueIntensity(value);
				if (intensity > maxIntensity) {
					maxPhero = phero;
					maxIntensity = intensity;
//					}
				}
			}
		}
		return maxPhero;
	}

	@SuppressWarnings("unchecked")
	@Override
	public void paintPatch(Patch p, int x, int y, int index) {
		Pheromone<Float> selectedPhero = pherosViewNode == null ? null
				: (Pheromone<Float>) getPheroWithMaxIntensity(index);
		if (selectedPhero != null) {
			float value = selectedPhero.get(index);
			if (value > 0.0001) {
				getGraphics().setFill(selectedPhero.getValueColor(value));
				getGraphics().fillRect(x, y, getCellSize(), getCellSize());
			}
		} else {
			super.paintPatch(p, x, y, index);
		}
	}

	/**
	 * Get the selected pheromones.
	 * 
	 * @return the selected pheromones
	 */
	public List<Pheromone<Float>> getSelectedPheros() {
		return selectedPherosCache;
	}

	/**
	 * Perform the selection of all pheromones in the view node.
	 */
	public void selectAllPheros() {
		pherosViewNode.selectAllPheros();
	}
}