# import numpy as np
# import os
#
# acc_path = os.path.join("output", "acc-loss_plot","66", "acc")
# loss_path = os.path.join("output", "acc-loss_plot","66", "loss")
#
# for file_name in os.listdir(acc_path):
#     print(file_name)
#
# for file_name in os.listdir(loss_path):
#     print(file_name)


import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# Set global font size and weight
mpl.rcParams['font.size'] = 40  # Change the 14 to any value you want
mpl.rcParams['font.weight'] = 'bold'  # Other options include 'normal', 'light', 'heavy'

cat = "lecture\\output_sgdvsadam"

acc_path = os.path.join(cat, "acc_train")
loss_path = os.path.join(cat, "acc_val")

# Define the file paths
acc_files = [os.path.join(acc_path, file_name) for file_name in os.listdir(acc_path)]
loss_files = [os.path.join(loss_path, file_name) for file_name in os.listdir(loss_path)]

# Load the acc data
acc_data = [pd.read_csv(file).assign(Model=os.path.basename(file).split('.')[0]) for file in acc_files]
acc_df = pd.concat(acc_data)

# Load the loss data
loss_data = [pd.read_csv(file).assign(Model=os.path.basename(file).split('.')[0]) for file in loss_files]
loss_df = pd.concat(loss_data)

# Plot the accuracy data
plt.figure(figsize=(30, 10))
plt.subplot(1, 2, 1)  # 1 row, 2 cols, subplot 1
for model, group in acc_df.groupby('Model'):
    plt.plot(group['Step'], group['Value'], label=model, linewidth=6, alpha=0.7)
plt.xlabel('Epoch', fontsize=40, fontweight='bold')  # Change the 16 to any value you want
plt.ylabel('Accuracy/train', fontsize=40, fontweight='bold')  # Change the 16 to any value you want
plt.legend(fontsize=40)  # Change the 14 to any value you want

# Plot the loss data
plt.subplot(1, 2, 2)  # 1 row, 2 cols, subplot 2
for model, group in loss_df.groupby('Model'):
    plt.plot(group['Step'], group['Value'], label=model, linewidth=6, alpha=0.7)
plt.xlabel('Epoch', fontsize=40, fontweight='bold')  # Change the 16 to any value you want
plt.ylabel('Accuracy/validation', fontsize=40, fontweight='bold')  # Change the 16 to any value you want
plt.legend(fontsize=40)  # Change the 14 to any value you want

plt.tight_layout()
plt.savefig(f"{cat}.png")
plt.show()
