package turtlekit.viewer.jfx;

import java.util.concurrent.CountDownLatch;

import javafx.animation.AnimationTimer;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;

/**
 * A window for a TKViewer
 * 
 * @author fab
 *
 */
public class AgentFxWindow extends javafx.stage.Stage {

    public static CountDownLatch latch = new CountDownLatch(1);
    private static AgentFxWindow window;

    private GraphicsContext gc;
    private JFXViewer myAgent;

    public AgentFxWindow(JFXViewer jfxViewer, int width, int height) {
	JFXManager.lock.lock();
	setTitle("FX Viewer");
	Group root = new Group();
	Canvas canvas = new Canvas(width, height);
	gc = canvas.getGraphicsContext2D();
	root.getChildren().add(canvas);
	setScene(new Scene(root));
	show();
	new AnimationTimer() {
	    @Override
	    public void handle(long now) {
		if (myAgent != null) {
		    myAgent.observe2();
		}
	    }
	}.start();
	window = this;
	myAgent = jfxViewer;
	JFXManager.fxDone.signal();
	JFXManager.lock.unlock();
   }
    
    public static AgentFxWindow getNewStage() {
       return window;
    }

    /**
     * @return the gc
     */
    public GraphicsContext getGc() {
	return gc;
    }

}
