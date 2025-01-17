
package turtlekit.launcher;

import java.io.File;
import java.lang.reflect.Method;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

import javafx.application.Application;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.ScrollPane;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;

public class TurtleKitLauncher extends Application {

	private List<Class<?>> findClassesWithMainMethod(String packageName) throws Exception {
		List<Class<?>> classesWithMain = new ArrayList<>();
		String path = packageName.replace('.', '/');
		URL resource = getClass().getClassLoader().getResource(path);
		if (resource == null) {
			return classesWithMain;
		}
		File directory = new File(resource.toURI());
		if (!directory.exists()) {
			return classesWithMain;
		}
		for (File file : directory.listFiles()) {
			if (file.isDirectory()) {
				classesWithMain.addAll(findClassesWithMainMethod(packageName + "." + file.getName()));
			} else if (file.getName().endsWith(".class")) {
				String className = packageName + '.' + file.getName().substring(0, file.getName().length() - 6);
				Class<?> clazz = Class.forName(className);
				try {
					Method mainMethod = clazz.getMethod("main", String[].class);
					if (mainMethod != null) {
						classesWithMain.add(clazz);
					}
				} catch (NoSuchMethodException e) {
					// No main method, ignore this class
				}
			}
		}
		return classesWithMain;
	}

	@Override
	public void start(Stage primaryStage) throws Exception {
		VBox root = new VBox();
		root.setSpacing(10);

		// Find all classes with a main method in the turtlekit packages
		List<Class<?>> agentClasses = findClassesWithMainMethod("turtlekit");

		for (Class<?> agentClass : agentClasses) {
			Button button = new Button("Launch " + agentClass.getSimpleName());
			button.setOnAction(_ -> launchModel(agentClass));
			root.getChildren().add(button);
		}

		ScrollPane scrollPane = new ScrollPane(root);
		scrollPane.setFitToWidth(true);

		Scene scene = new Scene(scrollPane, 300, 250);
		primaryStage.setTitle("TurtleKit Launcher");
		primaryStage.setScene(scene);
		primaryStage.show();
	}

	private void launchModel(Class<?> agentClass) {
		try {
			Method mainMethod = agentClass.getMethod("main", String[].class);
			mainMethod.invoke(null, (Object) new String[] {});
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public static void main(String[] args) {
		launch(args);
	}
}
