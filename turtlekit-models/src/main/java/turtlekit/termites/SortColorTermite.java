package turtlekit.termites;

import static javafx.scene.paint.Color.BLACK;
import static javafx.scene.paint.Color.CYAN;
import static javafx.scene.paint.Color.GREEN;
import static javafx.scene.paint.Color.MAGENTA;
import static javafx.scene.paint.Color.RED;
import static javafx.scene.paint.Color.YELLOW;

import java.util.List;
import java.util.Random;

import javafx.scene.paint.Color;
import turtlekit.kernel.TKEnvironment;

public class SortColorTermite extends Termite {

	private static List<Color> colors = List.of(MAGENTA, YELLOW, GREEN, CYAN);// ,BLUE);

	@Override
	protected void onActivation() {
		super.onActivation();
		getEnvironment().askPatchesOnStartup(p -> {
			p.setColor(prng().nextDouble() < chipsDensity ? colors.get(prng().nextInt(colors.size())) : BLACK);
		});
	}

	@Override
	public void findEmptyPatch() {
		if(getPatchColor() != getColor()) {
			if (getPatchColor() == BLACK) {
				setPatchColor(getColor());
				setColor(RED);
				changeNextBehavior("getAway");
				fd(getJump());
			} else {
				setColor(getPatch().swapColor(getColor()));
				changeNextBehavior("findNewPile");
				fd(getJump());
			}
		}
		else
			wiggle();
	}

	/** another one step behavior */
	@Override
	public void searchForChip() {
		wiggle();
		if (getPatchColor() != BLACK) {
			setColor(getPatch().swapColor(BLACK));
			fd(getJump());
			changeNextBehavior("findNewPile");
		}
	}

	/** another one step behavior */
	@Override
	public void findNewPile() {
		wiggle();
		if (getPatchColor() == getColor())
			changeNextBehavior("findEmptyPatch");
	}

	public static void main(String[] args) {
		executeThisTurtle(5000);
	}
}
