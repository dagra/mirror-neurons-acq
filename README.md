# Mirror Neurons - What did i just do?
This is a python implementation of Bonaiuto and Arbib's scientific paper 
[Extending the mirror neuron system model, II: what did I just do? A new role for mirror neurons.](https://www.ncbi.nlm.nih.gov/pubmed/20217428)

It is an attempt to reproduce the results for teaching purposes.

## How to run
### Requirements
Requires **python 2.7+**.

The neural network simulating the mirror system is implemented with [pytorch](https://pytorch.org/).

Additional libraries can be install through pip, using


```
    pip install -r requirements.txt
```

### Run
The parameters of the Augmented Competitive Queue (ACQ) can be set in the
file **acq_parameters.py**.

To run one simulation, set the parameters (with or without a mirror system)
in the file **simulate_one.py** and execute it

```
    python simulate_one.py
```

To run many simulations with varying number of irrelevant actions, set
the corresponding parameters in the file **simulate_many.py** and execute it

```
    python simulate_many.py
```

## Sample Results
### Single Simulation
The figures bellow depict the desirability curves of the actions and the
trial lengths with and without the mirror system. 
The solid vertical line signifies the the moment of the lesion, whereas the
dashed vertical line is the time of recovery, i.e. the time when the _"cat"_
found the new optimal strategy.

With the mirror system | Without the mirror system
:--------------------: | :-----------------------:
![desirability per action](../master/images/ms_desirabilities_all.png) | ![desirability per action](../master/images/no_ms_desirabilities_all.png)
![desirability per action](../master/images/ms_desirabilities.png) | ![desirability per action](../master/images/no_ms_desirabilities.png)
![trial length](../master/images/ms_lengths_all.png) | ![trial length](../master/images/no_ms_lengths_all.png)
![trial length](../master/images/ms_lengths.png) | ![trial length](../master/images/no_ms_lengths.png)
### Recovery
The figure bellow shows the mean trials (along with the standard error) required for recovery after the lesion as a function
of the number of irrelevant actions for an agent with the mirror system and without.

The main difference, in contrast to the paper, is that the agent without the mirror system cannot find easily the optimal
strategy even if only 1 irrelevant actions is available. But the important point, that as the number of irrelevant
actions increases the agent without the mirror system needs more trials for recovery, but the agent with the mirror system
is not affected.

![desirability per action](../master/images/recovery_rate.png)

## References

`Bonaiuto, J., & Arbib, M. A. (2010). Extending the mirror neuron system model, II: what did I just do? A new role for mirror neurons. Biological cybernetics, 102(4), 341-359.`
