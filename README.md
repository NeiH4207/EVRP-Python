# VRP-Project
VRP stands for Vehicle Routing Problem, which is a well-known optimization problem in the field of operations research and logistics. It involves determining the optimal routes for a fleet of vehicles to deliver goods or services to a set of customers while minimizing costs or maximizing efficiency.

| List of works | Status |
|---------------|--------|
| EVRP          |        |
| TSP           |        |
| CVRP          |        |
| ...           |        |

### Installation

1. Create environment
``` sh
conda create -n vrp-project python=3.11
```

2. Install libraries

``` sh
pip install .
```
3. Run test Greedy Search
``` sh
python examples/main.py  --dataset-path EVRP/benchmark-2019 -a HMAGS --result-path ./results/EVRP/HMAGS -r 10
```