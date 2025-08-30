import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def preprocess_data(df):
    # Drop rows with missing values
    df = df.dropna(axis=0)
    
    # Drop unnecessary columns
    X = df.drop(['#', 'video_id'], axis=1)
    
    # Encode target variable and ensure it's integer type
    X = X.copy()
    X.loc[:, 'claim_status'] = X['claim_status'].map({'opinion': 0, 'claim': 1}).astype(int)
    
    # Dummy encode categorical values
    X = pd.get_dummies(X, columns=['verified_status', 'author_ban_status'], drop_first=True)
    
    # Isolate target variable and ensure it's integer
    y = X['claim_status'].astype(int)
    X = X.drop(['claim_status'], axis=1)
    
    # Split the data
    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)
    
    # Text processing with CountVectorizer
    count_vec = CountVectorizer(ngram_range=(2, 3), max_features=15, stop_words='english')
    
    # Process training text
    count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()
    count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())
    X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)
    
    # Process validation text
    val_count_data = count_vec.transform(X_val['video_transcription_text']).toarray()
    val_count_df = pd.DataFrame(data=val_count_data, columns=count_vec.get_feature_names_out())
    X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True), val_count_df], axis=1)
    
    # Process test text
    test_count_data = count_vec.transform(X_test['video_transcription_text']).toarray()
    test_count_df = pd.DataFrame(data=test_count_data, columns=count_vec.get_feature_names_out())
    X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text']).reset_index(drop=True), test_count_df], axis=1)
    
    return X_train_final, X_val_final, X_test_final, y_train, y_val, y_test