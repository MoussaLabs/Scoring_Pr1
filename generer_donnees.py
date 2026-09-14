"""
Genere le fichier data/credit.csv.

Ce fichier a deja ete genere et est inclus dans le depot : tu n'as pas besoin
de relancer ce script. Il est la pour montrer d'ou viennent les donnees.

Les donnees imitent le jeu classique "German Credit" (1000 demandes de credit,
30 % de defauts de paiement), avec des noms de colonnes en francais.
"""

import numpy as np
import pandas as pd

# On fixe le hasard pour que le fichier soit toujours identique.
np.random.seed(42)

N = 1000

# --- Les modalites possibles de chaque variable, avec leur frequence ---------

modalites = {
    "compte_courant": (
        ["pas de compte", "solde negatif", "0 a 200 euros", "plus de 200 euros"],
        [0.40, 0.27, 0.27, 0.06],
    ),
    "historique_credit": (
        ["defauts passes", "retards passes", "credits en cours ok",
         "tous rembourses", "aucun credit"],
        [0.04, 0.09, 0.53, 0.29, 0.05],
    ),
    "epargne": (
        ["pas d epargne", "moins de 100 euros", "100 a 500 euros",
         "500 a 1000 euros", "plus de 1000 euros"],
        [0.19, 0.60, 0.10, 0.06, 0.05],
    ),
    "anciennete_emploi": (
        ["sans emploi", "moins de 1 an", "1 a 4 ans", "4 a 7 ans", "plus de 7 ans"],
        [0.06, 0.17, 0.34, 0.17, 0.26],
    ),
    "patrimoine": (
        ["immobilier", "assurance vie", "voiture", "aucun"],
        [0.28, 0.23, 0.33, 0.16],
    ),
    "logement": (
        ["locataire", "proprietaire", "loge gratuitement"],
        [0.18, 0.71, 0.11],
    ),
    "autres_credits": (
        ["credit dans une autre banque", "credit dans un magasin", "aucun"],
        [0.14, 0.05, 0.81],
    ),
    "objet_credit": (
        ["voiture neuve", "voiture occasion", "equipement maison", "tv hifi",
         "electromenager", "reparations", "etudes", "formation", "business", "autre"],
        [0.23, 0.10, 0.18, 0.28, 0.02, 0.05, 0.04, 0.01, 0.06, 0.03],
    ),
    "situation_familiale": (
        ["celibataire", "marie", "divorce", "veuf"],
        [0.40, 0.40, 0.14, 0.06],
    ),
}

donnees = {}
for nom, (valeurs, probas) in modalites.items():
    donnees[nom] = np.random.choice(valeurs, size=N, p=probas)

# --- Les variables numeriques ------------------------------------------------

donnees["age"] = np.clip(np.random.gamma(9, 3.9, N), 19, 75).round().astype(int)
donnees["montant_credit"] = np.clip(np.random.lognormal(7.8, 0.72, N), 250, 20000).round().astype(int)
donnees["duree_mois"] = np.clip(np.random.gamma(3, 7, N), 4, 72).round().astype(int)
donnees["taux_effort"] = np.random.choice([1, 2, 3, 4], N, p=[0.14, 0.23, 0.16, 0.47])
donnees["anciennete_logement"] = np.random.choice([1, 2, 3, 4], N, p=[0.13, 0.31, 0.15, 0.41])
donnees["nb_credits_en_cours"] = np.random.choice([1, 2, 3], N, p=[0.63, 0.34, 0.03])
donnees["personnes_a_charge"] = np.random.choice([0, 1, 2], N, p=[0.60, 0.25, 0.15])

df = pd.DataFrame(donnees)

# --- On fabrique le defaut de paiement --------------------------------------
# Chaque modalite ajoute ou retire du risque. Plus le total est eleve,
# plus la probabilite de defaut est grande.

risque = -1.05
risque += df["compte_courant"].map({
    "solde negatif": 0.95, "0 a 200 euros": 0.45,
    "plus de 200 euros": -0.10, "pas de compte": -0.95})
risque += df["historique_credit"].map({
    "defauts passes": 1.05, "aucun credit": 0.85, "credits en cours ok": 0.10,
    "retards passes": -0.30, "tous rembourses": -0.75})
risque += df["epargne"].map({
    "moins de 100 euros": 0.40, "100 a 500 euros": 0.15, "500 a 1000 euros": -0.20,
    "plus de 1000 euros": -0.85, "pas d epargne": -0.25})
risque += df["anciennete_emploi"].map({
    "sans emploi": 0.45, "moins de 1 an": 0.30, "1 a 4 ans": 0.00,
    "4 a 7 ans": -0.35, "plus de 7 ans": -0.20})
risque += df["patrimoine"].map({
    "immobilier": -0.35, "assurance vie": -0.05, "voiture": 0.05, "aucun": 0.55})
risque += df["logement"].map({
    "locataire": 0.25, "proprietaire": -0.15, "loge gratuitement": 0.30})
risque += df["autres_credits"].map({
    "credit dans une autre banque": 0.45, "credit dans un magasin": 0.20, "aucun": -0.10})
risque += 0.022 * (df["duree_mois"] - 21)
risque += 0.45 * (np.log(df["montant_credit"]) - 7.8)
risque += -0.020 * (df["age"] - 35)
risque += 0.10 * (df["taux_effort"] - 3)
risque += np.random.normal(0, 0.55, N)   # part imprevisible

proba_defaut = 1 / (1 + np.exp(-risque))
df["defaut"] = (np.random.random(N) < proba_defaut).astype(int)

df.to_csv("data/credit.csv", index=False)
print("Fichier data/credit.csv cree :", df.shape[0], "lignes")
print("Taux de defaut :", round(df["defaut"].mean() * 100, 1), "%")
