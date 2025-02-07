These folders include all the data you will need.

CQA from https://www.tau-nlp.sites.tau.ac.il/commonsenseqa
These are multiple choice questions, the answer choices and, for test and dev, the answer keys.
CQA includes:
-Train file: train_rand_split.jsonl
-Validation file: dev_rand_split.json
-Test file: test_rand_split_no_answers.jsonl

CoS-E from https://github.com/salesforce/cos-e
These are human explanations for the multiple choice questions, as well as selected words which humans have designated as important to get to the right answer choice.
CoS-E includes:
-Train file: cose_train_v1.0.jsonl
-Train file (processed): cose_train_v1.0_processed.jsonl
-Validation file: cose_dev_v1.0.jsonl
-Validation file (processed): cose_dev_v1.0_processed.jsonl

Merged files
These can be created using parse-commonsenseQA_v1.0.py for train and validation and parse_commonsenseQAtest_v1.0.py using these console commands:

python <path to parse_commonsenseQA_v1.0.py or path to commonsenseQAtest_v1.0.py> <path to CQA-file> <if train or dev: path to CoS-E file> <path and name of the desired output file>

Note that parse_commonsenseQAtest_v1.0.py is made by us and includes assumptions as to what shape the test.csv file should look like. 
This could lead to errors further down the line.
Included are the data files we created using CQA and processed CoS-E files when applicable:

-Train file: train.csv
-Validation file: dev.csv
-Test file: test.csv

IMPORTANT: The merged file names should stay the same, as that is what further code expects.