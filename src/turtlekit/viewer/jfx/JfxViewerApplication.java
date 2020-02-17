package turtlekit.viewer.jfx;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.stage.Stage;


public class JfxViewerApplication extends Application {

    private static GraphicsContext gc;
    private static JfxDefaultViewer myAgent;

    @Override
    public void start(Stage primaryStage) throws Exception {
        primaryStage.setTitle("Drawing Operations Test");
        Group root = new Group();
        Canvas canvas = new Canvas(800, 800);
        gc = canvas.getGraphicsContext2D();
        root.getChildren().add(canvas);
        primaryStage.setScene(new Scene(root));
        primaryStage.show();
        

        new AnimationTimer() {

	    @Override
            public void handle(long now) {
//		    gc.fillOval(0, 0, random  .nextInt(50), random .nextInt(50));
        	if (myAgent != null) {
		    myAgent.observe2();
		}
            }
        }.start();
    }

    public static void main(String[] args) {
	JfxViewerApplication.launch(args);
    }

    
    /**
     * @return the gc
     */
    public static GraphicsContext getGc() {
        return gc;
    }

    /**
     * @param myAgent the myAgent to set
     */
    public static void setMyAgent(JfxDefaultViewer agent) {
	myAgent = agent;
    }

}
