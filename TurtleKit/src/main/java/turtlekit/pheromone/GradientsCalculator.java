package turtlekit.pheromone;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.IntStream;

public class GradientsCalculator {

	public static double getMaxDirectionRandomWhenEquals(AbstractPheromoneGrid<Float> phero, int index) {
		double max = Double.NEGATIVE_INFINITY;
		double maxDirection = 0;
		List<Double> maxCandidates = new ArrayList<>();
		int[] neighborsIndexes = phero.getNeighborsIndexes();
		for (double direction = 0; direction < 360; direction += 45) {
			float neighborValue = phero.get(neighborsIndexes[index]);
			if (neighborValue >= max) {
				if (neighborValue > max) {
					max = neighborValue;
					maxCandidates.clear();
					maxDirection = direction;
				}
				maxCandidates.add(direction);
			}
			index++;
		}
		if (maxCandidates.size() > 1) {
			return maxCandidates.get(phero.prng().nextInt(maxCandidates.size())).intValue();
		} else {
			return (int) maxDirection;
		}
	}

	public static double getMaxDirectionNoRandom(AbstractPheromoneGrid<Float> phero, int index) {
		double max = Double.NEGATIVE_INFINITY;
		double maxDirection = 0;
		int[] neighborsIndexes = phero.getNeighborsIndexes();
		for (double direction = 0; direction < 360; direction += 45) {
			float neighborValue = phero.get(neighborsIndexes[index]);
			if (neighborValue > max) {
				max = neighborValue;
				maxDirection = direction;
			}
			index++;
		}
		return (int) maxDirection;
	}

	public static double getMinDirectionRandomWhenEquals(AbstractPheromoneGrid<Float> phero, int index) {
		List<Double> minCandidates = new ArrayList<>();
		double min = Double.POSITIVE_INFINITY;
		double minDirection = 0;
		int[] neighborsIndexes = phero.getNeighborsIndexes();
		for (double direction = 0; direction < 360; direction += 45) {
			float neighborValue = phero.get(neighborsIndexes[index]);
			if (neighborValue <= min) {
				if (neighborValue < min) {
					min = neighborValue;
					minCandidates.clear();
					minDirection = direction;
				}
				minCandidates.add(direction);
			}
			index++;
		}
		if (minCandidates.size() > 1) {
			return minCandidates.get(phero.prng().nextInt(minCandidates.size())).intValue();
		} else {
			return (int) minDirection;
		}
	}

	public static double getMinDirectionNoRandom(AbstractPheromoneGrid<Float> phero, int index) {
		double min = Double.POSITIVE_INFINITY;
		double minDirection = 0;
		int[] neighborsIndexes = phero.getNeighborsIndexes();
		for (double direction = 0; direction < 360; direction += 45) {
			float neighborValue = phero.get(neighborsIndexes[index]);
			if (neighborValue < min) {
				min = neighborValue;
				minDirection = direction;
			}
			index++;
		}
		return (int) minDirection;
	}

	static void computeGradients(AbstractPheromoneGrid<Float> phero, double[] maxGradients, double[] minGradients) {
		IntStream.range(0, maxGradients.length).parallel().forEach(i -> {
			double max = Double.NEGATIVE_INFINITY;
			double min = Double.POSITIVE_INFINITY;
			int[] neighborsIndexes = phero.getNeighborsIndexes();
			List<Double> maxCandidates = new ArrayList<>();
			List<Double> minCandidates = new ArrayList<>();
			int index = i * 8;
			for (double direction = 0; direction < 360; direction += 45) {
				float neighborValue = phero.get(neighborsIndexes[index]);
				if (neighborValue >= max) {
					if (neighborValue > max) {
						max = neighborValue;
						maxCandidates.clear();
					}
					maxCandidates.add(direction);
				}
				if (neighborValue <= min) {
					if (neighborValue < min) {
						min = neighborValue;
						minCandidates.clear();
					}
					minCandidates.add(direction);
				}
				index++;
			}
			if (maxCandidates.size() > 1) {
				maxGradients[i] = maxCandidates.get(phero.prng().nextInt(maxCandidates.size()));
			} else {
				maxGradients[i] = max;
			}
			if (minCandidates.size() > 1) {
				minGradients[i] = minCandidates.get(phero.prng().nextInt(minCandidates.size()));
			} else {
				minGradients[i] = min;
			}
		});
	}

}
