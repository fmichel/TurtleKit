/*
 * Copyright or © or Copr. Fabien Michel, Olivier Gutknecht, Jacques Ferber (1997)

fmichel@lirmm.fr
olg@no-distance.net
ferber@lirmm.fr

This software is a computer program whose purpose is to
provide a lightweight Java API for developing and simulating Multi-Agent Systems (MAS).

This software is governed by the CeCILL-C license under French law and
abiding by the rules of distribution of free software.  You can  use,
modify and/ or redistribute the software under the terms of the CeCILL-C
license as circulated by CEA, CNRS and INRIA at the following URL
"http://www.cecill.info".

As a counterpart to the access to the source code and  rights to copy,
modify and redistribute granted by the license, users are provided only
with a limited warranty  and the software's author,  the holder of the
economic rights,  and the successive licensors  have only  limited
liability.

In this respect, the user's attention is drawn to the risks associated
with loading,  using,  modifying and/or developing or reproducing the
software by the user in light of its specific status of free software,
that may mean  that it is complicated to manipulate,  and  that  also
therefore means  that it is reserved for developers  and  experienced
professionals having in-depth computer knowledge. Users are therefore
encouraged to load and test the software's suitability as regards their
requirements in conditions enabling the security of their systems and/or
data to be ensured and,  more generally, to use and operate it in the
same conditions as regards security.

The fact that you are presently reading this means that you have had
knowledge of the CeCILL-C license and that you accept its terms.
 */
package turtlekit.kernel;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.logging.Level;

import picocli.CommandLine.Command;
import picocli.CommandLine.ITypeConverter;
import picocli.CommandLine.Option;

/**
 * @author Fabien Michel
 *
 */
@Command(name = "TurtleKit", version = "TK 4", description = "Lightweight ALife platform")
class TKCommandLine implements Callable<Integer> {

	@Option(names = "--tkLogLevel", converter = LevelConverter.class, description = "TK kernel log level")
	Level tkLogLevel;

	@Option(names = "--width", description = "world width")
	int width = 400;

	@Option(names = "--cuda", description = "cuda mode")
	boolean cuda = false;

	@Option(names = "--height", description = "world height")
	int height = 400;

	@Option(names = { "-t", "--turtles" }, arity = "1", description = "launch turtles on startup")
	List<String> turtles = Collections.emptyList();

//	@Option(names = { "-ui", "--uiParameterClass"}, arity = "1", description = "classes to add to viewers")
//	List<String> uiClasses = Collections.emptyList();

//	@Option(names = { "-v", "--viewers" }, arity = "1", defaultValue = "turtlekit.viewer.TKViewer", description = "launch viewers on startup")
//	List<String> viewers = List.of(TKViewer.class.getName());

	@Option(names = { "-pc", "--patchClass" }, arity = "1", defaultValue = "turtlekit.kernel.Patch", description = "patch class")
	String patchClass = Patch.class.getName();

//	@Option(names = { "-env", "--envClass" }, arity = "1", description = "env class")
//	String envClass = TKEnvironment.class.getName();

	/**
	 * 
	 */
	public TKCommandLine() {

	}

	@Override
	public Integer call() throws Exception {
		return 0;
	}

	/**
	 * Converter for log levels args
	 */
	static class LevelConverter implements ITypeConverter<Level> {

		public Level convert(String value) {
			return Level.parse(value);
		}
	}

}
