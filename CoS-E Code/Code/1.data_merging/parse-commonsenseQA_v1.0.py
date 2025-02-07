import jsonlines
import sys
import csv

#Merging tool aiming to combine CQA and CoS-E

expl = {}
#sys.argv[2] are the open-ended answers the researchers themselves have gathered (cos-e dataset)
#The JSONL file format offers some convenience when handling these sorts of lists. 
with open(sys.argv[2], 'rb') as f:
    #For example, each line in a JSONL file is its own object, which means we can easy iterate through them like we do here:
    for item in jsonlines.Reader(f):
        #Also, several bits of the JSONL file are encoded as keys: For example the "id" key, as well as the "explanation" and "open-ended" key. 
        #This offers us easy access to each segment of each line, which we can now place into the expl list for later use
        expl[item['id']] = item['explanation']['open-ended']



#sys.argv[1] are the questions from the Common Question Answering (CQA) dataset.
with open(sys.argv[1], 'rb') as f:
    #sys.argv[3] is the csv file we later want as output.
    #Note: Forcing this as utf-8 and newline='' is our addition to the code. 
    #Otherwise, this code throws up an error / creates wrong files on Windows machines. 
    with open(sys.argv[3],'w', encoding='utf-8', newline='') as wf:
        #The csv file is created here
        wfw = csv.writer(wf,delimiter=',',quotechar='"')
        #writes column names into the csv
        wfw.writerow(['id','question','choice_0','choice_1','choice_2','label','human_expl_open-ended'])
        #For each item in the CQA dataset, label it with a certain number depending on the answer chosen
        for item in jsonlines.Reader(f):
            label = -1
            if(item['answerKey'] == 'A'):
                label = 0
            elif(item['answerKey'] == 'B'):
                label = 1
            elif(item['answerKey'] == 'C'):
                label = 2
            #lastly, we write a new row into the csv containing plenty of information.
            wfw.writerow([item['id'], #unique identifier
                          item['question']['stem'], #The question from CQA
                          item['question']['choices'][0]['text'], #The options from the CQA multiple choice for this particular question
                          item['question']['choices'][1]['text'],
                          item['question']['choices'][2]['text'],
                          label, #The correct answer's number
                          expl[item['id']]]) #The explanation from the cos-e which we have gathered earlier
            
        #After this loop is through, we have a finished CSV file!
