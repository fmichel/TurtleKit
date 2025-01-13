/*******************************************************************************
 * TurtleKit 3 - Agent Based and Artificial Life Simulation Platform
 * Copyright (C) 2011-2014 Fabien Michel
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/
package turtlekit.pheromone;

import java.util.Map;

import org.controlsfx.control.PropertySheet;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.Node;
import javafx.scene.control.Label;
import javafx.scene.control.ListView;
import javafx.scene.layout.VBox;
import madkit.gui.PropertySheetFactory;
import turtlekit.viewer.jfx.FXPheroViewer;

public class PheroView extends VBox {

//	@UIProperty(category = "Phero view", displayName = "red canal")
	private double red;

	/**
	 * @return the red
	 */
	public double getRed() {
		return red;
	}

	/**
	 * @param red the red to set
	 */
	public void setRed(double red) {
		this.red = red;
	}

	// private DefaultBoundedRangeModel green;
//	private DefaultBoundedRangeModel blue;
	private Pheromone<Float> myPhero;
	private FXPheroViewer fxPheroViewer;
	// private Environment env;
//	private JPanel evapPanel;
//	private JPanel diffusionPanel;

	public PheroView(Pheromone<Float> selectedPheromone) {
		myPhero = selectedPheromone;

		PropertySheet sheet = PropertySheetFactory.getSheet(selectedPheromone);
		getChildren().add(sheet);
//		PropertySheet sheet2 = ParametersSheetFactory.getParametersSheet(this);
//		getChildren().add(sheet2);
//		PropertySheet sheet3 = ParametersSheetFactory.getParametersSheet(new TestObjectProperty());
//		getChildren().add(sheet3);

	}

	public PheroView(FXPheroViewer fxPheroViewer, Map<String, Pheromone<Float>> pheromones) {
		this.fxPheroViewer = fxPheroViewer;
		ObservableList<String> l = FXCollections.observableArrayList(pheromones.keySet());
		ListView<String> listView = new ListView<>(l);
		getChildren().add(listView);
//		pheromones.entrySet().stream().forEach(e -> {
//			getChildren().add(new Label(e.getKey()));
//			getChildren().add(ParametersSheetFactory.getParametersSheet(e.getValue()));
//		});
		listView.getSelectionModel().selectedItemProperty().addListener((observable, oldValue, newValue) ->{
			fxPheroViewer.setSelectedPheromone(newValue);
			getChildren().removeLast();
			getChildren().removeLast();
			addPheroParamView(pheromones, newValue);
		});
		addPheroParamView(pheromones, pheromones.keySet().stream().findFirst().orElse(null));
	}

	/**
	 * @param pheromones
	 * @param newValue
	 */
	private void addPheroParamView(Map<String, Pheromone<Float>> pheromones, String newValue) {
		getChildren().add(new Label(newValue));
		getChildren().add(PropertySheetFactory.getSheet(pheromones.get(newValue)));
	}

	public void addNode(Node n) {
		getChildren().add(n);
	}

}
