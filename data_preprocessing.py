from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    X = df.drop('species', axis=1)
    y = df['species']

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return X, y
