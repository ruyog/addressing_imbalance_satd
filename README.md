# addressing_imbalance_satd
This repository contains the source code for evaluating different balancing schemes for SATD classification in SE domain.

Exeuction Configuration:
Set the path for input, output, log, trained model and the text transformer file.
Input - Input csv file containing labeled source code comments.
Output - Metrics - Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity, Geometric-Mean for each balancing scheme in latex (.tex) files and csv files containing the test data along with the output prediction probablity.
Log - for logging
Model Path - Location for storing the trained model with respective balancing scheme
Encoder Path - Location for storing the TF-IDF transformer which transforms the raw input text into vectorized format for the machine learning algorithm.

output_path=<>
log_path=<>
model_path=<>
encoder_path=<>
input_file=<>

Exeuction command:
python3 <python_source_file> <ml_model>
Example:
For Logistic Regression, Random Forest, XGBoost
python3 bal_td_within_proj.py "lr"
python3 bal_td_within_proj.py "rf"
python3 bal_td_within_proj.py "gb"


