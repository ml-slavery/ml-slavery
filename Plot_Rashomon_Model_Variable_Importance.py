import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
rashomon_threshold = 0.2340852835055713
path = "Outputs/Rashomon_Set/Model_Variable_Importance/"
df = pd.read_csv(path+"Model_Variable_Importance_Threshold_{}.csv".format(rashomon_threshold))
df.set_index('Unnamed: 0', drop=True, inplace=True)

rash_df = pd.read_csv("Outputs/Rashomon_Set/Rash_Results_all_{}.csv".format(rashomon_threshold))

n_vars=10
for col in df:
    MAE = round(float(col),4)
    rash_df["MAE_round"]=round(rash_df["MAE"], 4)
    model_name = rash_df["Model"][rash_df["MAE_round"] == MAE]
    features = rash_df["Features"][rash_df["MAE_round"] == MAE]
    title = str(features.values[0]) + " : " + str(model_name.values[0]) + ", MAE = " + str(MAE)
    print(col)
    print(title)
    model_df = pd.DataFrame(df[col])
    model_df["Feature"] = model_df.index
    model_df.sort_values(by=str(col), inplace=True, ascending=False)
    model_df=model_df[0:n_vars]

    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.set(font_scale=1)

    color = cm.viridis(np.linspace(.6, 1.2, 20))
    chart = sns.barplot(
        data=model_df,
        x=str(col),
        y="Feature",
        palette=color,
        orient='h'
    )
    chart.set(ylabel='Feature', xlabel='Permutation Importance', title=title)
    plt.subplots_adjust(bottom=0.2, left=0.2)
    plt.tight_layout()
    plt.savefig(str(path) + "Feature_Importance_{}.png".format(title))
    plt.clf()

# plot all together:
df = df.transpose()
df["MAE"] = pd.to_numeric(df.index, errors='coerce')
df["MAE_round"] = round(df["MAE"], 4)

rash_df["MAE_round"] = round(rash_df["MAE"],4)
df_all = rash_df.merge(df, on="MAE_round")

# melt:
df_all["Model Type"] = df_all["Features"].astype(str) + " : " + df_all["Model"].astype(str) + ", MAE = " + df_all["MAE_round"].astype(str)
df_all.drop(["Unnamed: 0", "MAE_x", "MAE_y", "Model", "Features", "MAE_round"], inplace=True, axis=1)
melt = pd.melt(df_all, id_vars="Model Type", var_name="Feature", value_name="Permutation Importance")

plt.figure(figsize=(30, 30))
sns.set(style="whitegrid")
sns.set(font_scale=1)

chart = sns.barplot(
    data=melt,
    x="Permutation Importance",
    y="Feature",
    palette="Set2",
    orient='h',
    hue="Model Type"
)
chart.set(ylabel='Feature', xlabel='Permutation Importance')
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.tight_layout()
plt.savefig(str(path) + "Feature_Importance_All.png")
plt.clf()

# choose only top 20 features from best model:
top_model_MAE = '0.2272672655393896'
df2 = df.transpose()
# df2.rename(columns={top_model_MAE:"Best_Model"}, inplace=True)
df2.sort_values(by=top_model_MAE, inplace=True, axis=0, ascending=False)
df2 = df2[0:25]
df2.to_csv(path + "Rashomon_feature_importance_sorted.csv")

df2 = df2.transpose()
df2["MAE"] = pd.to_numeric(df2.index, errors='coerce')
df2["MAE_round"] = round(df2["MAE"], 4)

df_all = rash_df.merge(df2, on="MAE_round")

# melt:
df_all["Model Type"] = df_all["Features"].astype(str) + " : " + df_all["Model"].astype(str) + ", MAE = " + df_all["MAE_round"].astype(str)
df_all.drop(["Unnamed: 0", "MAE_x", "MAE_y", "Model", "Features", "MAE_round"], inplace=True, axis=1)
melt = pd.melt(df_all, id_vars="Model Type", var_name="Feature", value_name="Permutation Importance")

plt.figure(figsize=(30, 30))
sns.set(style="whitegrid")
sns.set(font_scale=1)
sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=3)
chart = sns.barplot(
    data=melt,
    x="Permutation Importance",
    y="Feature",
    palette="Set2",
    orient='h',
    hue="Model Type"
)
chart.set(ylabel='Feature', xlabel='Permutation Importance')
plt.subplots_adjust(bottom=0.2, left=0.2)
plt.tight_layout()
plt.savefig(str(path) + "Feature_Importance_Best_Model_top25.png")
plt.clf()




print("done!")