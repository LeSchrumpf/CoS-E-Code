These folders include all the code you will need to run to ideally replicate the author's results.
There are added comments by us and the authors which aim to clarify the code. 
Further, some fixes and attempted fixes have been added as well. At times, unused functions have been removed to create a better readable code.

The code includes the following steps:

STEP 1:

Data merging: Merging CQA and CoS-E datasets.
-use parse-commonsenseQA_v1.0.py to merge training and dev sets.
-use parse-commonsenseQAtest_v1.0.py to create a test file.
How to use: 

-Adapt and run the following command in your console:

python <path to parse_commonsenseQA_v1.0.py or path to commonsenseQAtest_v1.0.py> <path to CQA-file> <if train or dev: path to CoS-E file> <path and name of the desired output file>

Note that parse_commonsenseQAtest_v1.0.py is made by us and includes assumptions as to what shape the test.csv file should look like. 
This could lead to errors further down the line.

STEP 2:

Explanation generation: Here, we train a GPT-2 model to generate Language Model explanations based on CQA and human explanations. 
These explanations will later provide the BERT model with additional context during its training, aiming to improve its performance.
How to use: 
(Example usage)
-Adapt and run the following command in your console:

python <path to train_commonsenseqa_v1.0.py> --model_name openai-gpt --do_train --do_eval --do_test --data <path to directory containing train.csv, dev.csv, test.csv> --output_dir <path to where you want model checkpoints and results>

This will train a GPT-2 model checkpoint called pytorch_model.bin 
As well as preds_explain_predict.txt, which are the model's predictions on questions in dev.csv
and test_explain_predict.txt, which are the model's predictions on questions in test.csv (This will probably contain only repeated predictions. Check comments in the test section of the code for approaches to fix this.)

Take a look at the parsing section in the code to get an idea as to what additional arguments you can pass when running the script.
You can adjust hyperparameters that way, for example. 

You can also load the model checkpoints, by commenting and un-commenting indicated lines in the code and running something like this:
python <path to train_commonsenseqa_v1.0.py> --model_name openai-gpt --do_eval --do_test --data <path to directory containing train.csv, dev.csv, test.csv> --output_dir <path to the MODEL CHECKPOINT!>

Make sure the right model checkpoint is contained within the output_dir. 
And that you do not run --train, or the model checkpoint will be overwritten.
It is therefore advisable to backup your output folder.

Lastly, you will have to "mis-use" the script to get model predictions for train.csv as well. This can be done by renaming train.csv" to dev.csv and dev.csv to train.csv in your data folder.
Then, you run the model in evaluation mode, loading the model you trained earlier. 
This, again, is based on the assumption that this is the intended use of this code, as there is no other obvious way to get LM explanations for train.csv, as the authors have done in their paper. 

STEP 3:

This code is added by us and therefore works a bit differently than the other scripts.
It copies and pastes the LM explanations generated in STEP 2 to the merged data files we got from STEP 1.
You will have to open the script in your scripting programm of choice and follow the instructions in the comments before running it.
You will be left with updated_train.csv, updated_dev.csv, and possibly even updated_test.csv
However, you will only need to pass updated_train.csv and updated_dev.csv to the next step's script. 

If you found a good spot for them within your folders, rename them to train.csv, dev.csv and test.csv again.

STEP 4: 

Training and using the BERT classifier.

How to use:
(Example usage)
-Adapt and run the following command in your console:

python <path to run_commonsenseQA_expl_v1.0.py> --data_dir <path to where you have the updated .csv files from STEP 3> --bert_model bert-base-uncased --output_dir <where you want model checkpoints and other output to go> --do_lower_case --do_train --do_eval --do_select

This will train the BERT model using LM explanations, save a checkpoint in output_dir, as well as printing evaluation metrics.
--do_select determines whether we are using human explanations or LM explanations. Putting --do_select in the command in the console will use LM explanations, and leaving it out will use human ones. 

Omitting -do_train, you can load a model checkpoint in much the same way as in STEP 2. 
However, you will have to comment and un-comment indicated lines in the python script, as well as placing the model checkpoint into the data_dir. 
The output_dir needs to be empty. 



Conclusion:
Following all of these steps will enable you to reproduce some of the results discussed in the author's paper. 
However, there are issues that stem from test.csv being unusable in some steps, which makes some parts difficult to replicate with the code at hand.

 

