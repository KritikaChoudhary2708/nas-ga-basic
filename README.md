Setup and Installation-->
1. Prerequisites
   -   Python 3.x
   -   PyTorch
   -   CUDA (optional, for GPU acceleration)

2. Installation
   -   Clone or download the repository.
   -   Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Execution

   To initiate the Neural Architecture Search, run the following command:

   ```bash
   python3 nas_run.py
   ```

Hyperparameters changed to increase excecution file:
   -   Population size: 6
   -   Number of generations: 3
   -   Mutation rate: 0.1
   -   Crossover rate: 0.9
   -   train_subset : 3000
   -   val_subset : 500
   -   train_loader : 512
   -   val_loader : 512
   -   epochs : 25
   -   learning rate : 0.002
   -   patience : 5