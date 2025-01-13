package turtlekit.kernel.activator;

import madkit.simulation.scheduler.DateBasedDiscreteEventActivator;
import turtlekit.agr.TKOrganization;

public class DateBasedPheromoneActivator extends DateBasedDiscreteEventActivator {

	public DateBasedPheromoneActivator() {
		super(TKOrganization.MODEL_GROUP, TKOrganization.ENVIRONMENT_ROLE, "executePheromonesSequentialy");
	}

}
