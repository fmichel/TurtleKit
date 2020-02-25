package turtlekit.viewer.jfx;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import javafx.application.Application;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.stage.Stage;


public class JFXManager extends Application {

    private static JFXManager fxApplicationInstance;
    private static AgentFxWindow newWindow;
    public static final CountDownLatch latch = new CountDownLatch(1);
    static Lock lock = new ReentrantLock();
    static Condition fxDone = lock.newCondition();

    private GraphicsContext gc;
    private JFXViewer myAgent;
    
    public static JFXManager waitForFxInitialization() {
        try {
            latch.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return fxApplicationInstance;
    }

    public static void setFxApplicationInstance(JFXManager fxa) {
	fxApplicationInstance = fxa;
        latch.countDown();
    }
    
    
    public JFXManager() {
	setFxApplicationInstance(this);
    }
    
    @Override
    public void start(Stage primaryStage) throws Exception {
        primaryStage.setTitle("TK desktop to be done...");
        Group root = new Group();
        Canvas canvas = new Canvas(800, 800);
        gc = canvas.getGraphicsContext2D();
        root.getChildren().add(canvas);
        primaryStage.setScene(new Scene(root));
//        primaryStage.show();
//        new AnimationTimer() {
//	    @Override
//            public void handle(long now) {
//        	if (myAgent != null) {
//		    myAgent.observe2();
//		}
//            }
//        }.start();
    }
    
    public static AgentFxWindow createNewWindow(JFXViewer jfxViewer, int width, int height) {
	javafx.application.Platform.runLater(() -> new AgentFxWindow(jfxViewer,width,height));
	lock.lock();
	try {
	    fxDone.await(1,TimeUnit.SECONDS);
	}
	catch(InterruptedException e) {
	    e.printStackTrace();
	}
	lock.unlock();
 	newWindow = AgentFxWindow.getNewStage();
	return newWindow;
    }
    
    /**
     * @return the gc
     */
    public GraphicsContext getGc() {
        return gc;
    }
    public void setMyAgent(JFXViewer agent) {
	myAgent = agent;
    }
}
