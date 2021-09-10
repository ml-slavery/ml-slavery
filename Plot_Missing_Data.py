import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

miss = pd.read_csv("Data/Meta_Data/X_missing_vars.csv")
df = pd.read_csv("Data/Data.csv")

# Create index's:
Index = df[['Country', 'Data_year']]
Index_list = []
for i in range(0, 70):
    Index_list.append(str(Index.Country[i]) + '_' + str(Index.Data_year[i]))
    Index_list[i] = Index_list[i].replace(" ", "_")

miss.index = Index_list
miss.rename(index={'Democratic_Republic_of_the_Congo_2018':'Dem_Rep_Congo_2018'}, inplace=True)
df.index = Index_list
df.rename(index={'Democratic_Republic_of_the_Congo_2018':'Dem_Rep_Congo_2018'}, inplace=True)

# plot:
save_path = "Outputs/Meta/"
colours = ['white', 'lightblue']
f, ax = plt.subplots(figsize = (20,30))
sns.set_style("whitegrid")
plt.title('')
sns.heatmap(miss.isnull().transpose(), cmap=sns.color_palette(colours), cbar=False)
plt.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig(save_path+"Missing_data.pdf")

# ------------------------------------------------------
# % miss:

nas=miss.isnull().sum(axis=1)
nas=pd.DataFrame(nas)
nas.columns = ["Missing"]
nas.index=Index_list
nas["perc_miss"] = (nas["Missing"] / miss.shape[1]) * 100
nas.sort_values(by="perc_miss", axis=0, inplace=True)
print('done!')