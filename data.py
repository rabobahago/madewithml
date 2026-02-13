import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings; warnings.filterwarnings("ignore")
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split
# Data ingestion
DATASET_LOC = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/dataset.csv"
df = pd.read_csv(DATASET_LOC)
df.head()

print(df.columns)

df.tag.value_counts

#Split dataset
test_size = .2
train_df, val_df = train_test_split(df, \
    stratify=df.tag, test_size=test_size, \
        random_state=1234 )


train_df.tag.value_counts()
val_df.tag.value_counts() * int((1-test_size)/ test_size)

all_tag = Counter(df.tag)
all_tag.most_common()

tags, tag_counts = zip(*all_tag.most_common())
plt.figure(figsize=(10, 3))
ax =sns.barplot(x=list(tags), y=list(tag_counts))
ax.set_xticklabels(tags, rotation = 0, fontsize= 0)
plt.title('Tag Distribution', fontsize=14)
plt.ylabel('# of projects', fontsize=12)
plt.show()

# Most frequent tokens for each tag
tag = 'natural-language-processing'
plt.figure(figsize=(10, 6))
subset = df[df.tag == tag]
text = subset.title.values
cloud = WordCloud(stopwords=STOPWORDS, background_color='black', 
                  collocations=False, width=500, height=300).generate(' '.join(text))
plt.axis('off')
plt.imshow(cloud)