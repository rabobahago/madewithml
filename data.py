import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings; warnings.filterwarnings("ignore")
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split

import json 
import nltk
import re 
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
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

df['text'] = df['title'] + ' ' + df['description']
import nltk
nltk.download('stopwords')
STOPWORDS = stopwords.words("english")

def clean_text(text, stopwords=STOPWORDS):
    """Clean raw text string."""
    # Lower
    text = text.lower()

    # Remove stopwords
    pattern = re.compile(r'\b(' + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub('', text)

    # Spacing and filters
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)  # add spacing
    text = re.sub("[^A-Za-z0-9]+", " ", text)  # remove non alphanumeric chars
    text = re.sub(" +", " ", text)  # remove multiple spaces
    text = text.strip()  # strip white space at the ends
    text = re.sub(r"http\S+", "", text)  #  remove links

    return text

# Apply to dataframe
original_df = df.copy()
df.text = df.text.apply(clean_text)
print (f"{original_df.text.values[0]}\n{df.text.values[0]}")

# DataFrame cleanup
df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # drop cols
df = df.dropna(subset=["tag"])  # drop nulls
df = df[["text", "tag"]]  # rearrange cols
df.head()

# Label to index
tags = train_df.tag.unique().tolist()
num_classes = len(tags)
class_to_index = {tag: i for i, tag in enumerate(tags)}
class_to_index

from transformers import BertTokenizer
def tokenize(batch):
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))

tokenize(df.head(1))

def preprocess(df, class_to_index):
    """Preprocess the data."""
    df["text"] = df.title + " " + df.description  # feature engineering
    df["text"] = df.text.apply(clean_text)  # clean text
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")  # clean dataframe
    df = df[["text", "tag"]]  # rearrange columns
    df["tag"] = df["tag"].map(class_to_index)  # label encoding
    outputs = tokenize(df)
    return outputs

preprocess(df=train_df, class_to_index=class_to_index)