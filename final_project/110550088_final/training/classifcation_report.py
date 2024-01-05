import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# flycatcher = np.zeros([8, 8], dtype=np.float32) # 36~42
# gull = np.zeros([9, 9], dtype=np.float32) # 58~65
# kingfisher = np.zeros([6, 6], dtype=np.float32) # 78~82
# sparrow = np.zeros([22, 22], dtype=np.float32) #112~132
# tern = np.zeros([8, 8], dtype=np.float32) # 140~146
# vireo = np.zeros([8, 8], dtype=np.float32) # 150~156
# warbler = np.zeros([26, 26], dtype=np.float32) # 157~181
# woodpecker = np.zeros([7, 7], dtype=np.float32) # 186~191
# wren = np.zeros([26, 26], dtype=np.float32) # 192~198

species = [
    {
        "name": "flycatcher",
        "lb": 36,
        "ub": 42,
    },
    {
        "name": "gull",
        "lb": 58,
        "ub": 65,
    },
    {
        "name": "kingfisher",
        "lb": 78,
        "ub": 82,
    },
    {
        "name": "sparrow",
        "lb": 112,
        "ub": 132,
    },
    {
        "name": "tern",
        "lb": 140,
        "ub": 146,
    },
    {
        "name": "vireo",
        "lb": 150,
        "ub": 156,
    },
    {
        "name": "warbler",
        "lb": 157,
        "ub": 181,
    },
    {
        "name": "woodpecker",
        "lb": 186,
        "ub": 191,
    },
    {
        "name": "wren",
        "lb": 192,
        "ub": 198,
    },
]

# get unique labels



# parser for input csv file name
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="csv file name")
args = parser.parse_args()

# read csv file
df = pd.read_csv(args.input)
gt = pd.read_csv("val_gt.csv")

y_pred = df["label"].values
y = gt["label"].values


label = np.unique(gt["label"].values)
label = np.sort(label)


# print classification report
print(classification_report(y, y_pred, digits=4))
# print(df[df.label == "064.Ring_billed_Gull"])
# print(gt[gt.label == "064.Ring_billed_Gull"])
# save the report
with open(f"{args.input.replace('.csv', '')}.txt", "w") as f:
    f.write(classification_report(y, y_pred, digits=4))

# print confusion matrix
cm = confusion_matrix(y, y_pred)
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# using matplotlib to plot
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
# plt.xticks(tick_marks, ["0", "1"], rotation=45)
# plt.yticks(tick_marks, ["0", "1"])
plt.xlabel('Predicted label')
plt.ylabel('True label')
# save figure
plt.savefig(f"{args.input.replace('.csv', '')}.png")
exit(0)
# draw the confusion matrix for each species
for specie in species:
    for i in range(specie["lb"], specie["ub"]+1):
        print(f"{label[i]}")
    continue
    # get the confusion matrix for each specie
    cm = confusion_matrix(y, y_pred, labels=label[specie["lb"]:specie["ub"]+1])
    # using matplotlib to plot
    # reset plt
    plt.clf()
    # make plt bigger
    plt.figure(figsize=(10, 10), dpi=250)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # show number in each grid
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, "{:,}".format(cm[i, j]),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.title(f"Confusion Matrix for {specie['name']}")
    tick_marks = np.arange(specie["ub"]-specie["lb"]+1)
    plt.colorbar()
    plt.xticks(tick_marks, label[specie["lb"]:specie["ub"]+1], rotation=45, fontsize=5)
    plt.yticks(tick_marks, label[specie["lb"]:specie["ub"]+1], fontsize=5)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    # save figure
    folder = args.input.replace('.csv', '')
    os.makedirs("eval/"+folder, exist_ok=True)
    plt.savefig(f"eval/{folder}/{folder}_{specie['name']}.png")
    '''
    # calculate the accuracy for each specie
    # merge df 
    df_spec = pd.merge(df, gt, on="id")
    df_spec = df_spec[df_spec.label_y.isin(label[specie["lb"]:specie["ub"]+1])]
    y_pred_spec = df_spec["label_x"].values
    y_spec = df_spec["label_y"].values
    # print(y, y_pred)
    # print(f"{specie['name']}: ", end="")
    print(f"{classification_report(y_spec, y_pred_spec)}")
    '''