package turtlekit.pheromone;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ColorPicker;
import javafx.scene.control.ListCell;
import javafx.scene.control.ListView;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.SelectionMode;
import javafx.scene.control.TitledPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.util.Callback;
import madkit.gui.PropertySheetFactory;
import turtlekit.viewer.TKViewer;

public class PherosViewNode extends VBox {

	private Map<Pheromone<?>, TitledPane> pherosParameterPaneMap;
	private ListView<String> listView;

	public PherosViewNode(TKViewer fxPheroViewer) {
		pherosParameterPaneMap = new HashMap<>();
		Map<String, Pheromone<Float>> pherosMap = fxPheroViewer.getEnvironment().getPheromonesMap();
		Set<String> names = pherosMap.keySet();
		listView = new ListView<>(FXCollections.observableArrayList(names));
		getChildren().add(listView);
		listView.getSelectionModel().setSelectionMode(SelectionMode.MULTIPLE);
		listView.setMinHeight(400);
		VBox pheroCoeffs = new VBox();
		ScrollPane pherosParams = new ScrollPane(pheroCoeffs);
		listView.setCellFactory(new Callback<ListView<String>, ListCell<String>>() {
			@Override
			public ListCell<String> call(ListView<String> listView) {
				return new ListCell<String>() {
					private final CheckBox checkBox = new CheckBox();
					private final ColorPicker colorPicker = new ColorPicker();
					private final HBox hbox = new HBox(checkBox, colorPicker);

					{
						checkBox.selectedProperty().addListener((_, _, isSelected) -> {
							if (isSelected) {
								listView.getSelectionModel().select(getItem());
							} else {
								listView.getSelectionModel().clearSelection(listView.getItems().indexOf(getItem()));
							}
							updateParameters(pherosMap, pheroCoeffs);
						});

						colorPicker.setOnAction(event -> {
							String item = getItem();
							if (item != null) {
								Pheromone<Float> phero = pherosMap.get(item);
								if (phero != null) {
									phero.getColorModel().setBaseColor(colorPicker.getValue());
									setStyle("-fx-background-color: " + colorPicker.getValue().toString().replace("0x", "#"));
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
								colorPicker.setValue(phero.getColorModel().getBackgroundColor());
								setStyle("-fx-background-color: "
										+ phero.getColorModel().getBackgroundColor().toString().replace("0x", "#"));
							}
							setGraphic(hbox);
						} else {
							setText(null);
							setGraphic(null);
							setStyle("");
						}
					}
				};
			}
		});
		if (!listView.getItems().isEmpty()) {
			listView.getSelectionModel().select(0);
		}
		getChildren().add(pherosParams);
	}

	private void updateParameters(Map<String, Pheromone<Float>> pherosMap, VBox pheroCoeffs) {
		pheroCoeffs.getChildren().clear();
		for (String selectedItem : listView.getSelectionModel().getSelectedItems()) {
			Pheromone<?> phero = pherosMap.get(selectedItem);
			TitledPane pheroPane = pherosParameterPaneMap.computeIfAbsent(phero,
					_ -> PropertySheetFactory.getTitledPaneSheet(selectedItem, phero));
			pheroCoeffs.getChildren().add(pheroPane);
		}
	}

	public ObservableList<String> getSelectedPheromones() {
		return listView.getSelectionModel().getSelectedItems();
	}

	private boolean isSelected(String pheroName) {
		return listView.getSelectionModel().getSelectedItems().contains(pheroName);
	}

	public void selectAllPheros() {
		listView.getSelectionModel().selectAll();
	}

}
