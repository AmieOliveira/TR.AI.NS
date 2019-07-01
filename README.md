# TR.AI.NS System Simulation

This code corresponds to the simulation of the Automates Transportation 
System TR.AI.NS. Code should be run through the "Simulation" script. 
There are some necessary arguments, as will be shown below.

The basic command to run this simulation is given as:

```bash
    ./Simulation.py -m MAP_FILE [-nT NUMBER_OF_TRAINS]
                    [-fC FREQUENCY_OF_CLIENT] [-tS TOTAL_STEPS_RUN]
                    [-vS STEP_SPEED]
```

Where `MAP_FILE` is the path for the folder with the map information for the 
simulation. One can give any map made of strait lines, as long as it is given 
in the specified data format, which can be creating exporting the file 
`map_creator.numbers` to CSV files (without combining the tables into a single 
file).

The other arguments are optional and all have a default value if none is given.
Additionally one can use the help argument (`-h` or `--help`) to check all 
possible arguments and its descriptions.

* `NUMBER_OF_TRAINS` is the number of transportation units one wants to have in 
their simulation. 

* `FREQUENCY_OF_CLIENT` is a parameter to make the apparition of
clients (which is random) more or less frequent. Its default is set to 25 and 
the smaller this number the bigger the probability of a client popping up. If it
is set to 1 a client will pop up every step, and if it is set to a number over 
100 no client will ever pop up. 

* `TOTAL_STEPS_RUN` is a parameter to finish the 
simulation. If one is given the simulation finishes after the specified number 
of steps. Otherwise the simulation will finish after 10 clients are delivered.

* `STEP_SPEED` is the ratio of seconds per step the simulation will emulate.
Default is set to 1 s/step