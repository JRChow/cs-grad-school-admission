"""Horsing around with data."""
import pandas as pd
import matplotlib.pyplot as plt

cs_arw = pd.read_csv('dataset/gradcafe/cs_arw.csv', header=None)

cs_arw = cs_clean.dropna()

cs_clean.columns = ['rowid', 'uni_name',
                    'major', 'degree', 'season', 'decision', 'decision_method',
                    'decision_date', 'decision_timestamp', 'ugrad_gpa',
                    'gre_verbal', 'gre_quant', 'gre_writing', 'is_new_gre',
                    'gre_subject', 'status', 'post_data', 'post_timestamp',
                    'comments']

plt.plot(cs_clean['gre_quant'], cs_clean['gre_verbal'], 'bo')
plt.show()

# with open('dataset/gradcafe/train_x.csv') as train_x_f:
#     train_x_reader = csv.reader(train_x_f)
#     for row in train_x_reader:
#         print(len(row))
#         # print(', '.join(row))
