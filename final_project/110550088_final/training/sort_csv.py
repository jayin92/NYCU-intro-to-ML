import pandas as pd

files = [
    "eval_ori.csv", 
    "eval_wo_comb.csv",
    "eval_high_temp.csv",
    "eval_high_temp_wo_comb.csv",
    "eval_ori_no_crop.csv"
]

for file in files:
    df = pd.read_csv(file)
    df = df.sort_values(by=["id"])
    df.to_csv(file, index=False)