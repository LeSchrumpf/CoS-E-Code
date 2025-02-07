import jsonlines
import sys
import csv

#This is a variation of the parse-commonsenseQA script. The aim of this code snippet is to convert the test_rand_split_no_answers.jsonl into a csv file of a particular shape, as used by the train_commonsenseqa_v1.1.py script

expl = {}
with open(sys.argv[1], 'rb') as f:
    with open(sys.argv[2],'w', encoding='utf-8', newline='') as wf:
        wfw = csv.writer(wf,delimiter=',',quotechar='"')
        wfw.writerow(['id','question','choice_0','choice_1','choice_2','label','human_expl_open-ended'])
        for item in jsonlines.Reader(f):
            question = item['question']['stem']
            choices = [choice['text'] for choice in item['question']['choices']]
            label = -1 #this is what the other scripts expect when testing
            while len(choices) < 5:
                choices.append("")
            wfw.writerow([item['id'], 
                          question,
                          *choices[:3],
                          label, 
                          None]) #putting "none" here is an assumption. It could very well require something else to place into the empty explanations
            
