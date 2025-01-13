package turtlekit.langtonAnts;

import static javafx.scene.paint.Color.BLACK;
import static javafx.scene.paint.Color.WHITE;

import turtlekit.kernel.DefaultTurtle;

public class LangtonAnt extends DefaultTurtle {

    @Override
    protected void onActivation() {
	super.onActivation();
	changeNextBehavior("doIt");
	home();
	fd(0.5f);
	setHeading(90);
	fd(0.5f);
    }

    public void doIt() {
	if (getPatchColor() == BLACK) {
	    setPatchColor(WHITE);
	    turnLeft(90);
	}
	else {
	    setPatchColor(BLACK);
	    turnRight(90);
	}
	fd(1);
    }

    public static void main(String[] args) {
	executeThisTurtle(1
			,"--width","500"
			,"--height","500"
			);

//		Option.envDimension.toString(), "1000,1000", 
//		Option.renderingInterval.toString(), "550", //TODO
//		Option.startSimu.toString());
    }

}
