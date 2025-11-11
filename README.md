# Diverse-BBO: Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering
This project provides the code and technical details of our method "Latent Space Reverse Engineering" (LSRE) for generating better MetaBBO training problem set (Diverse-BBO), which has been recently accpeted by AAAI 2026 as a Poster paper.\
For the problem instances in Diverse-BBO, we recommend you to read their mathematical formulations in [this pdf file](https://github.com/MetaEvo/Diverse-BBO/blob/main/DiverseBBO.pdf) before you use them.\
For the technical details of LSRE, we provide the **Appendix.pdf** mentioned in our paper in [this pdf file](https://github.com/MetaEvo/Diverse-BBO/blob/main/Appendix.pdf).

## Citation
The PDF version of the paper is available [here](https://arxiv.org/abs/2509.15810). If you find our Diverse-BBO useful, please cite it in your publications or projects.
```latex
@article{wang2025instance,
  title={Instance Generation for Meta-Black-Box Optimization through Latent Space Reverse Engineering},
  author={Wang, Chen and Ma, Zeyuan and Cao, Zhiguang and Gong, Yue-Jiao},
  journal={arXiv preprint arXiv:2509.15810},
  year={2025}
}
```


## Requirements
You can install all of dependencies of LSRE and Diverse-BBO via the command below.
```bash
conda new -n Diverse_BBO
conda activate Diverse_BBO
pip install -r requirements.txt
```
## Quick Start
You can easily load the all 256 Diverse-BBO function instances by the following code:
```python
from problem.SOO.self_generated.Symbolic_bench_numpy.Symbolic_bench_Dataset \
   import Diverse_BBO_Dataset
import numpy as np

# All 256 function instances
func_list,_ = Diverse_BBO_Dataset.get_datasets(upperbound=5, # x is bounded in [-5,5]
                                             train_batch_size = 1,  # default batch size set to 1
                                             get_all = True)

# # Batch
# func_list,_ = Diverse_BBO_Dataset.get_datasets(upperbound=5, # x is bounded in [-5,5]
#                                              train_batch_size = config.train_batch_size,
#                                              get_all = True)

## Train-Test Split Option
# train_set,test_set = Diverse_BBO_Dataset.get_datasets(upperbound=5,    
#                                                     train_batch_size=config.train_batch_size,
#                                                       test_batch_size=config.test_batch_size,
#                                                         )
```
All 256 instances of the Diverse-BBO functions are loaded into the `func_list` by batch, and you can access each function by index or iteration:
```python
# access Function 1 
function = func_list[0]

# access Functions by iteration:
for function in func_list:
   print(f"This is Function {function.problemID}")
   # Your code ...

# # Batch
# # access Function 1 in Batch 1
# batch_function = func_list[0]
# function = batch_function[0]

# # access Functions by iteration:
# for batch_function in func_list:
#    for function in batch_function:
#       print(f"This is Function {function.problemID}")
#       # Your code ...
```
Each function instance has some key variables and methods:
```python
## Variable
function.problemID    # Function Instance's Number
function.lb           # X's Lower Bound
function.ub           # X's Upper Bound
function.dim          # Best Evaluate Dimension (From Cross-Dimension Local Search Strategy)

## Method
function.func()      # Evaluation Function Interface
```
Therefore, we can use the following procedure to evaluate the fitness value:
```python
num_points = 10000
x = np.random.rand(num_points,function.dim) * (func.ub - func.lb) + func.lb
# fitness value
y = function.func(x)
```


Additionally, for researchers interested in the specific mathematical formulations of function instances within Diverse-BBO, we provide complete mathematical expressions for all 256 function instances in [DiverseBBO.pdf](https://github.com/MetaEvo/Diverse-BBO/blob/main/DiverseBBO.pdf).

## Instance Generation

We provide [function_generator_oneray.py](https://github.com/MetaEvo/Diverse-BBO/blob/main/function_generator_oneray.py) for single-layer parallel LSRE-based function instance search. You can quickly evaluate a single target function in parallel using the following command:
```bash
python function_generator_oneray.py --func_id 1 (can be any target sample point ID)  
```
For detailed parameter settings, please refer to [function_generator_oneray.py](https://github.com/MetaEvo/Diverse-BBO/blob/main/function_generator_oneray.py).
Additionally, if you are interested in the implementation details of LSRE, please examine the code in `gplearn_final_AE_ray_best_ela`. We also retain most of the latent space construction workflow examples in [train_AE.py](https://github.com/MetaEvo/Diverse-BBO/blob/main/train_AE.py) and [sample_problems.py](https://github.com/MetaEvo/Diverse-BBO/blob/main/sample_problems.py).

## Train
The training can be easily implemented by using Diverse-BBO for the MetaBBO approaches mentioned in the paper with the command below.
```bash
python main.py --train --train_agent GLEET_Agent --train_optimizer GLEET_Optimizer --train_batch_size 2  --train_mode multi --end_mode epoch  --problem Diverse-BBO --no_tb True --max_epoch 100
```
Where --train_agent and --train_optimizer should be passed with the corresponding MetaBBO method names (DE_DDQN, LDE, SYMBOL, GLEET) and their suffixes. For more adjustable settings, please refer to `trainer.py` and `config.py` for details.

The saved checkpoints will be saved to `./agent_model`, the file structure is as follow:
```
agent_model
|--train
   |--MetaBBO's name_Agent
      |---YYYYMMDDTHHmmSS_Benchmark_Dimension
         |----checkpoint0.pkl
         |----checkpoint1.pkl
         |----...
```

## Test
The test process can be easily activated via the command below.
```bash
python main.py --test --problem Diverse-BBO --agent_load_dir [The checkpoint saving directory, default to be "./agent_model"] --agent_for_cp [The run name of the target MetaBBO model] --l_optimizer_for_cp [The run name of the target MetaBBO model]_Optimizer --test_run 51
```
For example, for testing the model with run_name "GLEET_DiverseBBO" stored in "./agent_model", the command is:
```bash
python main.py --test --problem Diverse-BBO --agent_load_dir agent_model --agent_for_cp GLEET_DiverseBBO --l_optimizer_for_cp GLEET_Optimizer --test_run 51
```
The test result will be saved to `./output/test`, the file structure is as follow:
```
output
|--test
   |--YYYYMMDDTHHmmSS_Benchmark_Dimension
      |---test.pkl

```


