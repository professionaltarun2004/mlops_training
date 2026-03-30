import mlflow.sklearn

def predict(text):
    model = mlflow.sklearn.load_model(
        "models:/TextClassifier@production"
    )
    prediction = model.predict([text])
    return prediction[0]

if __name__=="__main__":
    #run_id="a5215d40b2ed4d6fabc42972e851c2b4"
    result=predict("zomato order")
    print("Prediction: ",result)