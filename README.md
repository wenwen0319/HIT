# HIT

# Requirements
* `python = 3.7`, `PyTorch = 1.4`, please refer to their official websites for installation details.
* Other dependencies:
```{bash}
pandas==0.24.2
tqdm==4.41.1
numpy==1.16.4
scikit_learn==0.22.1
matploblib==3.3.2
```
Refer to `environment.yml` for more details.

# Run the code
## Preprocess the dataset
#### Option 1: Use our data

Preprocess dataset
```{bash}
mkdir processed
python preprocess.py -d NDC-substances
```

This code is to preprocess the dataset, including zero-padding the node features and edge features.

#### Option 2: Use your own data
First download the dataset, for example https://www.cs.cornell.edu/~arb/data/
```{bash}
unzip dataset.zip
mkdir processed
python preprocess.py -d <dataset>
```

After preprocessing the dataset, we can run the code for three different questions.
## For Q1 type prediction
The task aims to solve the Q1 in the paper. What type of high-order interaction will most likely appear among 𝑢, 𝑣, 𝑤 within (𝑡, 𝑡 + 𝑇_𝑤]?

```{bash}
python main.py -d NDC-substances
```

The output will be in the log file. We will both report the AUC and the confusion matrix.

## For Q2 time prediction
The task aims to solve the Q2 in the paper. For a triplet ({𝑢, 𝑣 },𝑤, 𝑡 ), given an interaction pattern in {Wedge, Triangle, Closure}, when will 𝑢, 𝑣,𝑤 first form such
a pattern?

```{bash}
python main.py -d NDC-substances --time_prediction --time_prediction_type <time_prediction_type>
```
## Optional arguments
```{txt}
    --time_prediction_type For interpretation, we have 3 tasks. 1: Closure; 2: Triangle; 3: Wedge;  Default 0 means no time_prediction
```

The output will be in the ./time_prediction_output/<dataset>_<time_prediction_type>.txt.
We report the NLL loss and MSE for training, validating, and testing sets.


## For Q3 interpretation
```{bash}
python main.py -d NDC-substances --interpretation --interpretation_type 1
```

## Optional arguments
```{txt}
    --interpretation_type: Interpretation type: For interpretation, we have 3 tasks. 1: Closure vs Triangle; 2: Closure + Triangle vs Wedge; 3: Wedge and Edge; Default 0 means no interpretation
```
The output will be in the ./interpretation_output/<dataset>_<interpretation_type>.txt.
We report all the pattern we sample, with the times of each pattern appears in the first class and the total times it appears in both classes, and their ratio. We also report the mean score and variance of each pattern.


# Note

Finding edges, wedges, triangles, and closures process is in th utils.py. Since the finding process is time-consuming, the code will automatically save all the edges, wedges, triangles, closures in the saved_triplets. So the code doesn't need to run the process again in the future experiments.
