# V.A. - Automatic Transport System

This code corresponds to the simulation of the Automated and Distributed 
Transportation System called the VA Project. 

Code should be run through the "Simulation" script. There are some necessary 
arguments, as will be shown below. The basic command to run this simulation 
is given as:

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


## Versions

If you are running the code pay attention to the version you are using. Different
versions implement different features. 

If you use the latest v3 versions, you will be implementing the system in full 
dedicated mode, where one vehicle services only one client at a time. On v4 versions,
on the other hand, multiple clients can be picked up, as long as it does not add 
route deviations.

Enjoy!


[comment]: <> (TODO: Add the map format specifications!
    This simulation can be run with any map, as long as the pertinent 
    information is given in the correct format. This folder contains 
    a file called ")
    

#### Credits
The icons for the vehicles and users were obtained from the website 'www.flaticon.com' (accessed on 22th June 2019)
