package turtlekit.langtonAnts;

import java.awt.Color;

import turtlekit.kernel.Turtle;
import turtlekit.kernel.TurtleKit.Option;

public class RLR_LangtonAnt extends Turtle {
	
	@Override
	protected void activate() {
		super.activate();
		setNextAction("black");
		home();
		fd(0.5);
		setHeading(90);
		fd(0.5);
		setPatchColor(Color.BLACK);
		randomHeading();
		setColor(Color.BLACK);
	}

	public String black(){
		setPatchColor(Color.RED);
		turnRight(90);
		fd(1);
		return getColorName();
	}

	public String red(){
		setPatchColor(Color.GREEN);
		turnRight(90);
		fd(1);
		return getColorName();
	}

	public String green(){
		setPatchColor(Color.YELLOW);
		turnLeft(90);
		fd(1);
		return getColorName();
	}

	public String yellow(){
		setPatchColor(Color.RED);
		turnLeft(90);
		fd(1);
		return getColorName();
	}

	private String getColorName() {
		if(getPatchColor() == Color.RED)
			return "red";
		if(getPatchColor() == Color.GREEN)
			return "green";
		if(getPatchColor() == Color.YELLOW)
			return "yellow";
		if(getPatchColor() == Color.WHITE)
			return "white";
		return "black";
	}

	public String doIt(){
		final Color c = getPatchColor();
		if(c == Color.BLACK){
			setPatchColor(Color.RED);
			turnRight(90);
		}
		else if (c == Color.RED){
			setPatchColor(Color.YELLOW);
			turnRight(90);
		}
		else{
			setPatchColor(Color.GREEN);
			turnLeft(90);
		}
		fd(1);
		return "doIt";
	}

	public static void main(String[] args) {
		executeThisTurtle(5
				,Option.envDimension.toString(),"1000,1000"
				,Option.fastRendering.toString()
				,Option.startSimu.toString()
				);
	}

}
