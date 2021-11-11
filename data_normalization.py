from sklearn import preprocessing
import pandas as pd
audio = pd.read_csv("original_dataset")
names = audio.columns
d = preprocessing.normalize(audio, axis=0)
scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()
scaled_df.to_csv("modified_dataset.csv")
