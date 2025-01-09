import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


if __name__=='__main__':
   data = pd.read_csv('eval_seg_res.csv')
   fig = plt.figure()
   ax = fig.add_subplot(1, 2, 1)
   ax.set_xlabel('Dice')
   ax.set_ylabel('Value')
   dice_data=data[['dice_et','dice_tc','dice_wt']]
   dice_data=dice_data.rename(columns={"dice_et": "ET","dice_tc": "TC" , "dice_wt": "WT"})
   sns.boxplot(dice_data, ax=ax)

   ax = fig.add_subplot(1, 2, 2)
   ax.set_xlabel('HD95')
   ax.set_ylabel('Value')
   hd_data = data[['hd_et', 'hd_tc', 'hd_wt']]
   hd_data = hd_data.rename(columns={"hd_et": "ET", "hd_tc": "TC", "hd_wt": "WT"})

   print(hd_data.describe())
   sns.boxplot(hd_data, ax=ax)

   plt.show()
