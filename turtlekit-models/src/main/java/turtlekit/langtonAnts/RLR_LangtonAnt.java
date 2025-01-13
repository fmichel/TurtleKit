package turtlekit.langtonAnts;

import static javafx.scene.paint.Color.BLACK;
import static javafx.scene.paint.Color.GREEN;
import static javafx.scene.paint.Color.RED;
import static javafx.scene.paint.Color.WHITE;
import static javafx.scene.paint.Color.YELLOW;

import javafx.scene.paint.Color;
import turtlekit.kernel.DefaultTurtle;

public class RLR_LangtonAnt extends DefaultTurtle {
	
	@Override
	protected void onActivation() {
		super.onActivation();
		changeNextBehavior("black");
		home();
		fd(0.5);
		setHeading(90);
		fd(0.5);
		setPatchColor(BLACK);
		randomHeading();
		setColor(BLACK);
	}

	public void black(){
		setPatchColor(RED);
		turnRight(90);
		fd(1);
		changeNextBehavior(getColorName());
	}

	public void red(){
		setPatchColor(GREEN);
		turnRight(90);
		fd(1);
		changeNextBehavior(getColorName());
	}

	public void green(){
		setPatchColor(YELLOW);
		turnLeft(90);
		fd(1);
		changeNextBehavior(getColorName());
	}

	public void yellow(){
		setPatchColor(RED);
		turnLeft(90);
		fd(1);
		changeNextBehavior(getColorName());
	}

	private String getColorName() {
		if(getPatchColor() == RED)
			return "red";
		if(getPatchColor() == GREEN)
			return "green";
		if(getPatchColor() == YELLOW)
			return "yellow";
		if(getPatchColor() == WHITE)
			return "white";
		return "black";
	}

	public String doIt(){
		final Color c = getPatchColor();
		if(c == BLACK){
			setPatchColor(RED);
			turnRight(90);
		}
		else if (c == RED){
			setPatchColor(YELLOW);
			turnRight(90);
		}
		else{
			setPatchColor(GREEN);
			turnLeft(90);
		}
		fd(1);
		return "doIt";
	}

	public static void main(String[] args) {
		executeThisTurtle(5
				,"--width","500"
				,"--height","500"
				);
	}

}
