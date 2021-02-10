# HIT
Neural Predicting Higher-order Patterns in Temporal Networks

The dataset can be downloaded from https://www.cs.cornell.edu/~arb/data/
Unzip the dataset.

# Run the code
Preprocess dataset
```
python preprocess.py -d NDC-substances
```
The processed data will be saved in the ./processed file.


For Q1 type prediction
```
python main.py -d NDC-substances
```

The output will be in the log file.

For Q2 time prediction
```
python main.py -d NDC-substances --time_prediction --time_prediction_type 1
```

The output will be in the time_prediction_output.


For Q3 interpretation
```
python main.py -d NDC-substances --interpretation --interpretation_type 1
```

The output will be in the interpretation_output.

# Note

The code will save all the edges, wedges, triangles, closures in the saved_triplets. So the code doesn't need to run the process again in the future experiments.