from flask import Flask, jsonify, send_file
import pandas as pd
import mlflow.lightgbm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
import jinja2


app = Flask(__name__)

data = pd.read_csv(r"C:\Users\dmedc\Documents\DATA SCIENCE\PROJET\Projet 7\kernel\info_clients.csv").head(1000)
df = pd.read_csv(r"C:\Users\dmedc\Documents\DATA SCIENCE\PROJET\Projet 7\kernel\data.csv").head(1000)

## Préparation des données
train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
# Séparer les caractéristiques et la variable cible
feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_BUREAU','SK_ID_PREV','index']]    
X = train_df[feats]
# Remplacer les NaN par la moyenne
X = X.fillna(X.mean())     

## Charger le modèle enregistré
local_path = r"C:\Users\dmedc\Documents\DATA SCIENCE\PROJET\Projet 7\kernel\modele"
model = mlflow.lightgbm.load_model(local_path)

## Création d'un explainer SHAP
explainer = shap.TreeExplainer(model)
## Initialisation de shap.initjs()
#shap.initjs()
## Chemin d'accès
folder = r"C:\Users\dmedc\Documents\DATA SCIENCE\PROJET\Projet 7\api/image"

## Page d'accueil
@app.route('/', methods=['GET'])
def hello():
    return " Bienvenue à la société financière, nommée 'Prêt à dépenser'"

## Récupérer les ID des clients à partir de la colonne "id" de la DataFrame
@app.route('/clients', methods=['GET'])
def get_clients():
    client_ids = data['SK_ID_CURR'].tolist()
    return jsonify(client_ids[:1000])

## Afficher les infos importantes sur un client
@app.route('/client/<int:id>', methods=['GET'])
def get_client(id):
    if id in data['SK_ID_CURR'].tolist() :
        clientid = data.loc[data['SK_ID_CURR']== id, : ]
        client = clientid.to_dict('records')
        return jsonify(client)
    else:   
        return f"Le numéro n'existe pas dans la base de données" 

## Faire la prediction et afficher le resultat
@app.route('/predict/<int:id>', methods=['GET'])
def predict(id):
    client_info = X[X['SK_ID_CURR'] == id]
    client_info = client_info.drop('SK_ID_CURR', axis=1)
    client_info= client_info.replace([np.inf, -np.inf], 1e9)

    if client_info.empty:
        return jsonify({"error": "Client pas trouvé"}), 404
    prediction = model.predict_proba(client_info)
    prediction = prediction.tolist()[0]
    return jsonify({"prediction": (prediction[1])})

## Faire l'interpretation locale de la prédiction
@app.route('/interpretation/local/<int:id>', methods=['GET'])
def get_local_interpretation(id):
    client_data = X[X['SK_ID_CURR'] == id]
    client_data = client_data.drop('SK_ID_CURR', axis=1)
    client_data = client_data.replace([np.inf, -np.inf], 1e9)

    if client_data.empty:
        return jsonify({"error": "Client pas trouvé"}), 404
    
    # Calculer les Shapley
    shap_values = explainer.shap_values(client_data)
    # Visualisation de l'importance des fonctionnalités
    shap.summary_plot(shap_values, client_data, max_display=10, plot_type='bar')
    # Enregistrer le graphique
    plt.savefig(f'{folder}/local_interpretation_{id}.png')

    return send_file(f'{folder}/local_interpretation_{id}.png', mimetype='image/png') 


## Faire l'interpretation Global du modèle
@app.route('/interpretation/global', methods=['GET'])
def get_global_interpretation():
    sample_data = X
    sample_data = sample_data.drop('SK_ID_CURR', axis=1)
    sample_data = sample_data.replace([np.inf, -np.inf], 1e9)

    shap_values = explainer(sample_data)
    shap.summary_plot(shap_values, sample_data, max_display =10, plot_type="bar")
    # Enregistrer le diagramme en tant qu'image
    plt.savefig(f'{folder}/global_interpretation.png')

    return send_file(f'{folder}/global_interpretation.png', mimetype='image/png')


if __name__ == '__main__':
    app.run()