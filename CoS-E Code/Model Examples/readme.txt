This folder contains example outputs for both train_commonsense and run_commonsense.
Model checkpoints are sadly too large to be uploaded to github. 

The GPT example used the default hyperparameters and all the data files contained in the Merged folder in Data.
Additionally, the console output.txt contains the model's output manually copy-pasted from the console. This will not be created automatically. 
-test_explain_predict.txt includes an example as to how test.csv leads to looping predictions.


The BERT example is using the default hyperparameters, with human explanations provided during both training and testing. 
Tests were performed both on updated_dev.csv and updated_test.csv.
-eval_results shows results on dev
-eval_results_test shows results on test
