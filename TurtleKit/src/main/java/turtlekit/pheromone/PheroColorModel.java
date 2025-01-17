package turtlekit.pheromone;

import java.util.Objects;
import java.util.Random;

import javafx.scene.paint.Color;

public class PheroColorModel {

	public static PheroColorModel RED = new PheroColorModel(100, 100, 100, MainColor.RED);
	public static PheroColorModel BLUE = new PheroColorModel(100, 100, 100, MainColor.BLUE);
	public static PheroColorModel GREEN = new PheroColorModel(100, 100, 100, MainColor.GREEN);

	private MainColor mainColor;
	private Color baseColor;
	private static Random random = new Random();

	public enum MainColor {
		RED, GREEN, BLUE;
	}

	public PheroColorModel(Color baseColor, MainColor mainColor) {
		this.baseColor = Objects.requireNonNull(baseColor);
		this.mainColor = Objects.requireNonNull(mainColor);
	}

	public PheroColorModel() {
		this(Color.color(random.nextDouble(.7), random.nextDouble(.7), random.nextDouble(.7)),
				MainColor.values()[random.nextInt(0, 2)]);
	}

	public PheroColorModel(int i, int j, int k, MainColor mainColor) {
		this(Color.rgb(i, j, k), mainColor);
}

	<T> Color getColor(Pheromone<T> pheromone, T value) {
		double max = pheromone.getLogMaxValue();
		Float valueOf = Float.valueOf(value.toString());
		double mainC = (int) (Math.log10(valueOf + 1) / max);
		mainC += 100;
		if (mainC > 255)
			mainC = 255;
		mainC /= 255;
		switch (mainColor) {
		case RED:
			return Color.color(mainC, baseColor.getGreen(), baseColor.getBlue());
		case GREEN:
			return Color.color(baseColor.getRed(), mainC, baseColor.getBlue());
		case BLUE:
			return Color.color(baseColor.getBlue(), baseColor.getBlue(), mainC);
		default:
			return Color.color(baseColor.getRed(), baseColor.getGreen(), baseColor.getBlue());
		}
	}


	public MainColor getMainColor() {
		return mainColor;
	}

	public void setMainColor(MainColor mainColor) {
		this.mainColor = mainColor;
	}

	public Color getBaseColor() {
		return baseColor;
	}

	public void setBaseColor(Color baseColor) {
		this.baseColor = baseColor;
	}

	public Color getBackgroundColor() {
		double opacity = .7;
		switch (mainColor) {
		case RED:
			return Color.color(.7, baseColor.getGreen(), baseColor.getBlue(), opacity);
		case GREEN:
			return Color.color(baseColor.getRed(), .7, baseColor.getBlue(), opacity);
		case BLUE:
			return Color.color(baseColor.getBlue(), baseColor.getBlue(), .7, opacity);
		default:
			return Color.color(baseColor.getRed(), baseColor.getGreen(), baseColor.getBlue(), opacity);
		}
	}

	@Override
	public String toString() {
		return "PheroColorModel [mainColor=" + mainColor + ", baseColor=" + baseColor + "]";
	}
}
