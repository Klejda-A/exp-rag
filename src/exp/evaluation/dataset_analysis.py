import pandas
from datasets import Dataset
import re

print("start")

dataset = Dataset.from_file("data/chatrag_quac/test/data-00001-of-00002.arrow")
df = dataset.to_pandas()

# answers = []
# for i in range(len(df)):
#     answers.append(len((re.sub(' +', ' ', (df["answers"].iloc[i])[0])).split(" ")))

questions = []
for i in range(len(df)):
    questions.append(len(((df["messages"].iloc[i])[-1]["content"]).split(" ")))
    
print(questions)
print("max" + str(max(questions)))
print("min" + str(min(questions)))
print("avg" + str(sum(questions)/len(questions)))

print("end")