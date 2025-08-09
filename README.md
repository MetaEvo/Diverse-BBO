# Diverse-BBO
The code and technical details of our method "Latent Space Reverse Engineering" (LSRE) for generating better MetaBBO training problem set (Diverse-BBO). For the 256 problem instances in Diverse-BBO, we provide their mathematical formulations in [thid pdf file](https://github.com/MetaEvo/Diverse-BBO/blob/main/all_256_programs_expr.pdf).

## Requirements
You can install all of dependencies of LSRE via the command below.
```bash
pip install -r requirements.txt
```
## Quick Start
We provide the demo.py file which demonstrates how to use the 256 functions in Diverse-BBO for simple evaluation and important considerations. You can run the following command to view it:
```bash
python demo.py
```
For instructions on how to access specific functions within Diverse-BBO and construct custom datasets using Diverse-BBO, please carefully read the `demo.py` and the `Symbolic_bench_Dataset class` used in `utils.py`.

Additionally, for researchers interested in the specific mathematical formulations of function instances within Diverse-BBO, we provide complete mathematical expressions for all 256 function instances in `all_256_programs_expr.pdf`.

## Instance Generation

We provide `function_generator_oneray.py` for single-layer parallel LSRE-based function instance search. You can quickly evaluate a single target function in parallel using the following command:
```bash
python function_generator_oneray.py --func_id 1 (can be any target sample point ID)  
```
For detailed parameter settings, please refer to `function_generator_oneray.py`.
Additionally, if you are interested in the implementation details of LSRE, please examine the code in `gplearn_final_AE_ray_best_ela`. We also retain most of the latent space construction workflow examples in `train_AE.py` and `sample_problems.py`.

## Train
The training can be easily implemented using Diverse-BBO for the MetaBBO approaches mentioned in the paper with the command below.
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
python main.py --test --problem Diverse-BBO --agent_load_dir [The checkpoint saving directory, default to be "./agent_model"] --agent_for_cp [The run name of the target MetaBBO model] --l_optimizer_for_cp GLEET_Optimizer --test_run 51
```
For example, for testing the model with run_name "GLEET_BBOB" stored in "./agent_model", the command is:
```bash
python main.py --test --problem Diverse-BBO --agent_load_dir agent_model --agent_for_cp GLEET_BBOB --l_optimizer_for_cp GLEET_Optimizer --test_run 51
```
The test result will be saved to `./output/test`, the file structure is as follow:
```
output
|--test
   |--YYYYMMDDTHHmmSS_Benchmark_Dimension
      |---test.pkl

```

