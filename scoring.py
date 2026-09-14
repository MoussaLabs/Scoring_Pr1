"""
==============================================================
 SCORING DE CREDIT - Construction d'une grille de score
==============================================================

 Lancer avec :  python scoring.py

 Le script fait 8 etapes, dans l'ordre :
   1. Charger les donnees
   2. Separer apprentissage et test
   3. Decouper chaque variable en classes
   4. Calculer le WoE et l'IV de chaque classe
   5. Choisir les variables a garder
   6. Entrainer une regression logistique
   7. Transformer le modele en grille de points
   8. Mesurer la performance et faire les graphiques

 Tous les resultats sont ecrits dans le dossier resultats/.
==============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # permet de sauvegarder les graphiques sans ecran
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

# Couleurs utilisees dans tous les graphiques
BLEU = "#3b6fb6"
ORANGE = "#d97757"
VERT = "#1f9e8f"
GRIS = "#8a8780"


# ==============================================================
# LES FONCTIONS DONT ON AURA BESOIN
# ==============================================================

def decouper_en_classes(serie, nb_classes=4):
    """Decoupe une variable numerique en classes de taille egale.

    Exemple : l'age devient 4 tranches contenant chacune 25 % des clients.
    On utilise pd.qcut, qui decoupe selon les quantiles.
    """
    return pd.qcut(serie, nb_classes, duplicates="drop").astype(str)


def regrouper_petites_classes(serie, part_minimum=0.05):
    """Regroupe dans 'autres' les modalites qui concernent moins de 5 % des clients.

    Une classe avec 3 clients ne veut rien dire statistiquement : on la fusionne.
    """
    frequences = serie.value_counts(normalize=True)
    petites = frequences[frequences < part_minimum].index
    return serie.replace(list(petites), "autres")


def calculer_woe(classes, cible):
    """Calcule le WoE et l'IV pour une variable deja decoupee en classes.

    WoE (Weight of Evidence) = ln( % de bons clients / % de mauvais clients )
      - WoE positif  -> la classe est MOINS risquee que la moyenne
      - WoE negatif  -> la classe est PLUS risquee que la moyenne

    IV (Information Value) = somme des contributions de chaque classe.
      C'est le pouvoir de la variable a separer les bons des mauvais.
    """
    tableau = pd.DataFrame({"classe": classes, "defaut": cible})

    resume = tableau.groupby("classe").agg(
        effectif=("defaut", "size"),
        mauvais=("defaut", "sum"),
    )
    resume["bons"] = resume["effectif"] - resume["mauvais"]
    resume["taux_defaut"] = resume["mauvais"] / resume["effectif"]

    # Le +0.5 evite une division par zero si une classe n'a aucun defaut.
    part_bons = (resume["bons"] + 0.5) / (resume["bons"].sum() + 0.5 * len(resume))
    part_mauvais = (resume["mauvais"] + 0.5) / (resume["mauvais"].sum() + 0.5 * len(resume))

    resume["woe"] = np.log(part_bons / part_mauvais)
    resume["contribution_iv"] = (part_bons - part_mauvais) * resume["woe"]

    return resume.reset_index()


def gini(vrai, predit):
    """Le Gini mesure la qualite du modele. Entre 0 (nul) et 1 (parfait).

    En credit, un Gini entre 0.40 et 0.60 est un bon modele.
    """
    return 2 * roc_auc_score(vrai, predit) - 1


def ks(vrai, predit):
    """Le KS est l'ecart maximum entre les bons et les mauvais clients cumules."""
    faux_positifs, vrais_positifs, _ = roc_curve(vrai, predit)
    return max(vrais_positifs - faux_positifs)


# ==============================================================
# ETAPE 1 : CHARGER LES DONNEES
# ==============================================================
print("\n=== ETAPE 1 : chargement des donnees ===")

donnees = pd.read_csv("data/credit.csv")

print("Nombre de dossiers :", len(donnees))
print("Nombre de variables :", donnees.shape[1] - 1)
print("Taux de defaut :", round(donnees["defaut"].mean() * 100, 1), "%")
print("\nAttention : on modelise toujours le DEFAUT (defaut = 1),")
print("jamais le bon client. Sinon tous les signes s'inversent.")


# ==============================================================
# ETAPE 2 : SEPARER APPRENTISSAGE ET TEST
# ==============================================================
print("\n=== ETAPE 2 : apprentissage / test ===")

# On construit le modele sur 70 % des dossiers et on le juge sur les 30 % restants.
# Le modele ne doit JAMAIS voir les donnees de test pendant sa construction.
X = donnees.drop(columns=["defaut"])
y = donnees["defaut"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,          # garde le meme taux de defaut dans les deux groupes
    random_state=42,     # pour obtenir toujours le meme decoupage
)

print("Apprentissage :", len(X_train), "dossiers,", round(y_train.mean() * 100, 1), "% de defaut")
print("Test          :", len(X_test), "dossiers,", round(y_test.mean() * 100, 1), "% de defaut")


# ==============================================================
# ETAPE 3 et 4 : DECOUPER LES VARIABLES ET CALCULER LE WoE
# ==============================================================
print("\n=== ETAPES 3 et 4 : decoupage en classes et calcul du WoE ===")

variables_numeriques = ["age", "montant_credit", "duree_mois", "taux_effort",
                        "anciennete_logement", "nb_credits_en_cours", "personnes_a_charge"]
variables_texte = [c for c in X.columns if c not in variables_numeriques]

# Pour chaque variable on stocke : ses classes, et la correspondance classe -> WoE
tables_woe = {}        # le tableau detaille de chaque variable
correspondances = {}   # le dictionnaire {nom de la classe : valeur du WoE}
liste_iv = []          # l'IV de chaque variable

for variable in X.columns:

    # 3a. On transforme la variable en classes
    if variable in variables_numeriques:
        classes_train = decouper_en_classes(X_train[variable])
    else:
        classes_train = regrouper_petites_classes(X_train[variable])

    # 3b. On calcule le WoE de chaque classe
    table = calculer_woe(classes_train, y_train)
    iv = table["contribution_iv"].sum()

    tables_woe[variable] = table
    correspondances[variable] = dict(zip(table["classe"], table["woe"]))
    liste_iv.append({"variable": variable, "iv": iv})

# On classe les variables de la plus utile a la moins utile
resume_iv = pd.DataFrame(liste_iv).sort_values("iv", ascending=False)

print(resume_iv.to_string(index=False))
print("\nComment lire l'IV :")
print("  moins de 0.02 : la variable ne sert a rien")
print("  0.02 a 0.10   : faible")
print("  0.10 a 0.30   : moyen")
print("  plus de 0.30  : fort")

print("\nExemple de tableau WoE, variable 'compte_courant' :")
print(tables_woe["compte_courant"].to_string(index=False))


# ==============================================================
# ETAPE 5 : CHOISIR LES VARIABLES
# ==============================================================
print("\n=== ETAPE 5 : selection des variables ===")

variables_gardees = resume_iv[resume_iv["iv"] >= 0.02]["variable"].tolist()
variables_ecartees = resume_iv[resume_iv["iv"] < 0.02]["variable"].tolist()

print("Gardees  :", len(variables_gardees), "->", ", ".join(variables_gardees))
print("Ecartees :", len(variables_ecartees), "->", ", ".join(variables_ecartees))


# On remplace maintenant chaque variable par son WoE.
# C'est ce tableau de nombres qui sera donne a la regression logistique.

def transformer_en_woe(tableau):
    """Remplace chaque valeur par le WoE de sa classe."""
    resultat = pd.DataFrame(index=tableau.index)
    for variable in variables_gardees:
        if variable in variables_numeriques:
            classes = decouper_en_classes(tableau[variable])
        else:
            classes = regrouper_petites_classes(tableau[variable])
        # .fillna(0) : si une classe n'existait pas a l'apprentissage, WoE neutre
        resultat[variable] = classes.map(correspondances[variable]).fillna(0)
    return resultat


X_train_woe = transformer_en_woe(X_train)
X_test_woe = transformer_en_woe(X_test)


# ==============================================================
# ETAPE 6 : LA REGRESSION LOGISTIQUE
# ==============================================================
print("\n=== ETAPE 6 : regression logistique ===")

modele = LogisticRegression(max_iter=1000)
modele.fit(X_train_woe, y_train)

coefficients = pd.DataFrame({
    "variable": variables_gardees,
    "coefficient": modele.coef_[0].round(3),
})
print(coefficients.to_string(index=False))

print("\nControle important : tous les coefficients doivent etre NEGATIFS.")
print("Un WoE eleve = client sain, donc il doit FAIRE BAISSER le risque.")
if (modele.coef_[0] < 0).all():
    print("-> OK, tous les coefficients sont negatifs.")
else:
    mauvaises = coefficients[coefficients["coefficient"] > 0]["variable"].tolist()
    print("-> A verifier, coefficient positif pour :", ", ".join(mauvaises))

# Le modele donne une probabilite de defaut pour chaque dossier
proba_train = modele.predict_proba(X_train_woe)[:, 1]
proba_test = modele.predict_proba(X_test_woe)[:, 1]


# ==============================================================
# ETAPE 7 : LA GRILLE DE POINTS
# ==============================================================
print("\n=== ETAPE 7 : grille de points ===")

# On transforme les probabilites en un score lisible, entre 300 et 900 environ.
# Regle choisie : 600 points = 50 bons clients pour 1 mauvais,
#                 et +20 points = deux fois moins de risque.
PDO = 20               # "Points to Double the Odds"
SCORE_CIBLE = 600
ODDS_CIBLE = 50

facteur = PDO / np.log(2)
decalage = SCORE_CIBLE - facteur * np.log(ODDS_CIBLE)
nb_variables = len(variables_gardees)

lignes_grille = []
for i, variable in enumerate(variables_gardees):
    coefficient = modele.coef_[0][i]
    table = tables_woe[variable]
    for _, ligne in table.iterrows():
        points = -(coefficient * ligne["woe"] + modele.intercept_[0] / nb_variables)
        points = points * facteur + decalage / nb_variables
        lignes_grille.append({
            "variable": variable,
            "classe": ligne["classe"],
            "effectif": ligne["effectif"],
            "taux_defaut": round(ligne["taux_defaut"], 3),
            "woe": round(ligne["woe"], 3),
            "points": int(round(points)),
        })

grille = pd.DataFrame(lignes_grille)
print(grille.to_string(index=False))


def calculer_score(tableau_woe):
    """Transforme la sortie du modele en points de score."""
    log_odds = -modele.decision_function(tableau_woe)
    return (decalage + facteur * log_odds).round().astype(int)


score_train = calculer_score(X_train_woe)
score_test = calculer_score(X_test_woe)

# On range les clients en 5 classes de risque, des plus risques aux plus surs
bornes = np.quantile(score_test, [0.2, 0.4, 0.6, 0.8])
classes_risque = pd.cut(
    score_test,
    [-np.inf] + list(bornes) + [np.inf],
    labels=["E (tres risque)", "D", "C", "B", "A (tres sain)"],
)

tableau_risque = pd.DataFrame({"classe": classes_risque, "defaut": y_test.values})
tableau_risque = tableau_risque.groupby("classe", observed=True).agg(
    effectif=("defaut", "size"),
    defauts=("defaut", "sum"),
)
tableau_risque["taux_defaut"] = (tableau_risque["defauts"] / tableau_risque["effectif"]).round(3)
tableau_risque["decision"] = ["refus", "refus", "etude manuelle", "accord", "accord"]

print("\nClasses de risque (sur l'echantillon de test) :")
print(tableau_risque.reset_index().to_string(index=False))


# ==============================================================
# ETAPE 8 : PERFORMANCE ET GRAPHIQUES
# ==============================================================
print("\n=== ETAPE 8 : performance ===")

performance = pd.DataFrame({
    "echantillon": ["apprentissage", "test"],
    "gini": [round(gini(y_train, proba_train), 3), round(gini(y_test, proba_test), 3)],
    "ks": [round(ks(y_train, proba_train), 3), round(ks(y_test, proba_test), 3)],
    "auc": [round(roc_auc_score(y_train, proba_train), 3),
            round(roc_auc_score(y_test, proba_test), 3)],
})
print(performance.to_string(index=False))
print("\nSeuls les chiffres du TEST comptent pour juger le modele.")

# --- Graphique 1 : l'IV de chaque variable ---
plt.figure(figsize=(7, 5))
a_tracer = resume_iv.sort_values("iv")
plt.barh(a_tracer["variable"], a_tracer["iv"], color=BLEU)
plt.axvline(0.02, color=GRIS, linestyle=":")
plt.text(0.021, 0, " seuil 0.02", color=GRIS, fontsize=8)
plt.xlabel("Information Value")
plt.title("Quelles variables separent le mieux bons et mauvais payeurs ?")
plt.tight_layout()
plt.savefig("resultats/1_information_value.png", dpi=120)
plt.close()

# --- Graphique 2 : le WoE des 4 meilleures variables ---
meilleures = resume_iv.head(4)["variable"].tolist()
figure, sous_graphiques = plt.subplots(2, 2, figsize=(11, 7))
for axe, variable in zip(sous_graphiques.ravel(), meilleures):
    table = tables_woe[variable]
    couleurs = [VERT if v >= 0 else ORANGE for v in table["woe"]]
    axe.bar(range(len(table)), table["woe"], color=couleurs)
    axe.set_xticks(range(len(table)))
    axe.set_xticklabels([str(c)[:14] for c in table["classe"]], rotation=30,
                        ha="right", fontsize=8)
    axe.axhline(0, color=GRIS, linewidth=1)
    axe.set_title(variable + "  (IV = " + str(round(table["contribution_iv"].sum(), 3)) + ")")
    axe.set_ylabel("WoE")
figure.suptitle("Vert = classe peu risquee   |   Orange = classe risquee")
plt.tight_layout()
plt.savefig("resultats/2_woe.png", dpi=120)
plt.close()

# --- Graphique 3 : la courbe ROC ---
plt.figure(figsize=(6, 5.5))
for vrai, predit, couleur, nom in [
    (y_train, proba_train, BLEU, "Apprentissage"),
    (y_test, proba_test, ORANGE, "Test"),
]:
    fp, vp, _ = roc_curve(vrai, predit)
    plt.plot(fp, vp, color=couleur, linewidth=2,
             label=nom + " - Gini " + str(round(gini(vrai, predit), 3)))
plt.plot([0, 1], [0, 1], color=GRIS, linestyle="--", label="Modele au hasard")
plt.xlabel("Part des bons clients refuses a tort")
plt.ylabel("Part des mauvais clients detectes")
plt.title("Courbe ROC")
plt.legend()
plt.tight_layout()
plt.savefig("resultats/3_courbe_roc.png", dpi=120)
plt.close()

# --- Graphique 4 : le taux de defaut par classe de risque ---
plt.figure(figsize=(7, 4.5))
plt.bar(tableau_risque.index.astype(str), tableau_risque["taux_defaut"], color=BLEU)
for i, valeur in enumerate(tableau_risque["taux_defaut"]):
    plt.text(i, valeur + 0.01, str(int(valeur * 100)) + " %", ha="center", fontsize=9)
plt.ylabel("Taux de defaut observe")
plt.title("Taux de defaut par classe de risque (echantillon de test)")
plt.tight_layout()
plt.savefig("resultats/4_classes_de_risque.png", dpi=120)
plt.close()


# ==============================================================
# SAUVEGARDE DES RESULTATS
# ==============================================================
grille.to_csv("resultats/grille_de_score.csv", index=False)
resume_iv.to_csv("resultats/information_value.csv", index=False)
coefficients.to_csv("resultats/coefficients.csv", index=False)
performance.to_csv("resultats/performance.csv", index=False)
tableau_risque.reset_index().to_csv("resultats/classes_de_risque.csv", index=False)

print("\n=== Termine ===")
print("Resultats ecrits dans le dossier resultats/ :")
print("  grille_de_score.csv, information_value.csv, coefficients.csv,")
print("  performance.csv, classes_de_risque.csv")
print("  et 4 graphiques au format png")
