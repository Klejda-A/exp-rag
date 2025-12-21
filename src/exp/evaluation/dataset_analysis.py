import pandas
from datasets import Dataset
import re



def context_analysis(contexts, questions, answers, conversations, tokenizer):
    contexts_all = []
    contexts_u = set()
    questions_all = []
    answers_all = []
    conversations_all = []

    for i in range(len(contexts)):
        contexts_u.add(contexts[i].content)
        contexts_all.append(len(tokenizer.encode(contexts[i].content)))

    print(len(contexts_u))
    print("CONTEXTS:")
    print("len" + str(len(contexts_all)))
    print("min" + str(min(contexts_all)))
    print("max" + str(max(contexts_all)))
    print("avg" + str(sum(contexts_all)/len(contexts_all)))

    for i in range(len(questions)):
        questions_all.append(len(tokenizer.encode(questions[i])))
        if isinstance(answers[0], list):
            for j in answers[i]:
                answers_all.append(len(tokenizer.encode(j)))
        else:
            answers_all.append(len(tokenizer.encode(answers[i])))
        # conversations_all.append(conversations[i].count("User:"))

    print("QUESTIONS:")
    print("len" + str(len(questions_all)))
    print("min" + str(min(questions_all)))
    print("max" + str(max(questions_all)))
    print("avg" + str(sum(questions_all)/len(questions_all)))        

    print("ANSWERS:")
    print("len" + str(len(answers_all)))
    print("min" + str(min(answers_all)))
    print("max" + str(max(answers_all)))
    print("avg" + str(sum(answers_all)/len(answers_all)))

    # conv_sequence = []
    # for i in range(len(conversations_all)):
    #     if i > 0:
    #         if conversations_all[i] == 1:
    #             conv_sequence.append(conversations_all[i-1])
    # print("CONVERSATIONS:")
    # print("min" + str(min(conv_sequence)))
    # print("max" + str(max(conv_sequence)))
    # print("avg" + str(sum(conv_sequence)/len(conv_sequence)))


def conversation_sequence(conversations, dataset_name):
    conversations_all = []

    for i in range(len(conversations)):
        conversations_all.append(conversations[i].count("User:"))

    conv_sequence = []
    for i in range(len(conversations_all)):
        if i == (len(conversations_all) - 1):
            conv_sequence.append(conversations_all[i])
        if i > 0:
            if conversations_all[i] == 1:
                conv_sequence.append(conversations_all[i-1])
    print(conv_sequence)
    with open("data/output.txt", "a") as f:
        f.write("\n" + dataset_name + "\n")
        f.write(", ".join(map(str, conv_sequence)))


def analyze_dataset():

    dataset = Dataset.from_file("data/chatrag_quac/test/data-00001-of-00002.arrow")
    df = dataset.to_pandas()

    answers = []
    for i in range(len(df)):
        answers.append(len((re.sub(' +', ' ', (df["answers"].iloc[i])[0])).split(" ")))
    questions = []
    for i in range(len(df)):
        questions.append(len(((df["messages"].iloc[i])[-1]["content"]).split(" ")))
    print(questions)
    print("max" + str(max(questions)))
    print("min" + str(min(questions)))
    print("avg" + str(sum(questions)/len(questions)))