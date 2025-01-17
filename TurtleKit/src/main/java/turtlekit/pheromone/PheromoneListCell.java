
package turtlekit.pheromone;

import java.util.Map;

import javafx.scene.control.CheckBox;
import javafx.scene.control.ColorPicker;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import madkit.gui.PropertySheetFactory;

public class PheromoneListCell extends ListCell<String> {
	private final CheckBox checkBox = new CheckBox();
	private final ColorPicker colorPicker;
	private final HBox hbox;
	private final Map<String, Pheromone<Float>> pherosMap;
	private final ListView<String> listView;
	private final VBox pheroCoeffs;

	public PheromoneListCell(Map<String, Pheromone<Float>> pherosMap, ListView<String> listView, VBox pheroCoeffs) {
		this.pherosMap = pherosMap;
		this.listView = listView;
		this.pheroCoeffs = pheroCoeffs;
		colorPicker = new ColorPicker();
		hbox = new HBox(checkBox, colorPicker);

		checkBox.selectedProperty().addListener((obs, wasSelected, isSelected) -> {
			if (isSelected) {
				listView.getSelectionModel().select(getItem());
			} else {
				listView.getSelectionModel().clearSelection(listView.getItems().indexOf(getItem()));
			}
			updateParameters();
		});

		colorPicker.setOnAction(_ -> {
			String item = getItem();
			if (item != null) {
				Pheromone<Float> phero = pherosMap.get(item);
				if (phero != null) {
					phero.getColorModel().setBaseColor(colorPicker.getValue());
					setStyle("-fx-background-color: "
							+ phero.getColorModel().getBackgroundColor().toString().replace("0x", "#"));
				}
			}
		});
	}

	@Override
	protected void updateItem(String item, boolean empty) {
		super.updateItem(item, empty);
		if (item != null && !empty) {
			setText(item);
			Pheromone<Float> phero = pherosMap.get(item);
			if (phero != null) {
				checkBox.setSelected(listView.getSelectionModel().getSelectedItems().contains(item));
				colorPicker.setValue(phero.getColorModel().getBaseColor());
				setStyle(
						"-fx-background-color: " + phero.getColorModel().getBackgroundColor().toString().replace("0x", "#"));
			}
			setGraphic(hbox);
		} else {
			setText(null);
			setGraphic(null);
			setStyle("");
		}
	}

	private void updateParameters() {
		pheroCoeffs.getChildren().clear();
		for (String selectedItem : listView.getSelectionModel().getSelectedItems()) {
			Pheromone<?> phero = pherosMap.get(selectedItem);
			TitledPane pheroPane = PropertySheetFactory.getTitledPaneSheet(selectedItem, phero);
			pheroCoeffs.getChildren().add(pheroPane);
		}
	}
}
