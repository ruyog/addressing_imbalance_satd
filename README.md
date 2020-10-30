# addressing_imbalance_satd
This repository contains the source code for evaluating different balancing schemes for SATD classification in SE domain.

Configuration:
Set the path for input, output, log, trained model and the text transformer file.
output_path='/scratch/project_2002565/bal_td/cross_rslt_tables/'
log_path='/scratch/project_2002565/bal_td/cross_log/'
model_path="/scratch/project_2002565/bal_td/chk_pt/"
encoder_path="/scratch/project_2002565/bal_td/encoder/"
input_file='/scratch/project_2002565/bal_td/data/technical_debt_dataset.csv'

Exeuction command:
python3 <python_source_file> <ml_model>
Example:
For Logistic Regression, Random Forest, XGBoost
python3 bal_td_within_proj.py "lr"
python3 bal_td_within_proj.py "rf"
python3 bal_td_within_proj.py "gb"


