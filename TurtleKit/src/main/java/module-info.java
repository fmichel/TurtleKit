open module turtlekit.base {
	requires transitive madkit.base;
	requires transitive jcuda;
	requires info.picocli;
	requires java.desktop;
	requires jcuda.natives;
	requires javafx.graphics;
	requires net.jodah.typetools;
	requires javafx.controls;
	requires java.logging;
	requires javafx.base;

	exports turtlekit.kernel;
	exports turtlekit.cuda;
	exports turtlekit.kernel.activator;
	exports turtlekit.agr;
	exports turtlekit.pheromone;
	exports turtlekit.viewer.jfx;
	exports turtlekit.viewer;
}