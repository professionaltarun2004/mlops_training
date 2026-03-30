from src.data_loader import load_data
from src.preprocess import split_data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import yaml  

def run_pipeline():
    with open("config/config.yaml","r") as f:
        config=yaml.safe_load(f)
    
    #step 1 - load data
    data=load_data()

    #step 2 - preprocess
    X,y = split_data(data)

    #step 3 - vectorize
    vectorizer=CountVectorizer()
    X_vec=vectorizer.fit_transform(X)

    #step 4 - train
    model=LogisticRegression(
        random_state=config["model_params"]["random_state"]   
    )
    model.fit(X_vec,y)

    #step 5 - save
    with open(config["model_path"],"wb") as f:
        pickle.dump((model,vectorizer),f)
    
if __name__=="__main__":
    run_pipeline()