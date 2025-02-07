import pandas as pd

# File paths
csv_file = "" #put file path to test.csv or dev.csv here
txt_file = "" #the text file in your output directory. Should be named preds_explain_predict.txt or test_explain_predict.txt
output_csv = "..\\train_updated.csv"  # Where you want your new file to go and how you want to name it

# Step 1: Load dev.csv
df = pd.read_csv(csv_file)

# Step 2: Read preds_explain_predict3.txt (each line is an explanation)
with open(txt_file, "r", encoding="utf-8") as f:
    explanations = [line.strip() for line in f.readlines()]

# Step 3: Ensure the lengths match
# You might have to manually ensure this is the case... sorry!
if len(df) != len(explanations):
    raise ValueError(f"Mismatch: dev.csv has {len(df)} rows, but explanations file has {len(explanations)} lines.")

# Step 4: Append the explanations as a new column
df["lm_explanation"] = explanations

# Step 5: Save the updated CSV
df.to_csv(output_csv, index=False)

print(f"Updated CSV saved as {output_csv}")
