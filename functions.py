import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate


def lire_reseau_flot(filename):
    """
    Lit un fichier texte contenant la table d'un réseau de flot.

    Format attendu :
      - La première ligne contient un entier n (nombre de sommets).
      - Ensuite, n lignes, chacune avec n entiers séparés par des espaces, représentant la matrice des capacités.
      - (Optionnel) Si le fichier contient ensuite un autre entier n et n lignes avec n entiers, alors cette seconde matrice sera
        traitée comme la matrice des coûts.

    Exemple pour un flot max :
      5
      0 9 5 7 0
      0 0 6 0 6
      0 0 0 0 7
      0 0 0 0 2
      0 0 0 0 0

    Retourne un dictionnaire contenant :
      - 'n': nombre de sommets
      - 'capacites': matrice des capacités (numpy array n x n)
      - 'couts': matrice des coûts (numpy array n x n) si présente, sinon None.
    """
    with open(filename, 'r') as file:
        lignes = [l.strip() for l in file if l.strip()]

    n = int(lignes[0])
    capacites = []
    for i in range(1, n + 1):
        row = list(map(int, lignes[i].split()))
        if len(row) != n:
            raise ValueError("La matrice des capacités doit être de taille n x n.")
        capacites.append(row)
    capacites = np.array(capacites)

    flot = construire_matrice_flot(n)

    couts = None
    if len(lignes) > n + 1:
        try:
            n2 = int(lignes[n + 1])
            if n2 != n:
                raise ValueError("La taille de la matrice des coûts doit être égale à n.")
            couts = []
            for i in range(n + 2, n + 2 + n):
                row = list(map(int, lignes[i].split()))
                if len(row) != n:
                    raise ValueError("La matrice des coûts doit être de taille n x n.")
                couts.append(row)
            couts = np.array(couts)
        except Exception as e:
            couts = None

    return {"n": n, "capacites": capacites, "couts": couts, "flot": flot}


def mapping_noeuds(n):
    """
    Crée un dictionnaire de mapping des indices vers des étiquettes :
      - 0  -> "s" (source)
      - n-1 -> "t" (puits)
      - Pour 1 <= i <= n-2, i -> lettre a, b, c, …
    """
    mapping = {}
    mapping[0] = "s"
    mapping[n - 1] = "t"
    for i in range(1, n - 1):
        mapping[i] = chr(ord('a') + (i - 1))
    return mapping


def afficher_tableau(mat, titre="Tableau", fmt="grid"):
    """
    Affiche la matrice (numpy array) de manière soignée à l'aide de tabulate
    en utilisant des étiquettes de nœuds (s, a, b, …, t) pour les lignes et colonnes.
    """
    n = mat.shape[0]
    mapping = mapping_noeuds(n)
    headers = [mapping[i] for i in range(n)]
    index_labels = [mapping[i] for i in range(n)]
    tableau = []
    for i in range(n):
        ligne = []
        for j in range(n):
            val = mat[i][j]
            if val == float('inf'):
                ligne.append("∞")
            else:
                ligne.append(str(val))
        tableau.append(ligne)
    print(titre)
    print(tabulate(tableau, headers=headers, showindex=index_labels, tablefmt=fmt))


def construire_graphe_flot(reseau):
    """
    Construit et retourne un graphe orienté (networkx.DiGraph) à partir du dictionnaire 'reseau'.

    Les sommets sont étiquetés avec des lettres :
      - Le sommet 0 est la source, étiqueté "s"
      - Le sommet n-1 est le puits, étiqueté "t"
      - Les sommets intermédiaires reçoivent les lettres "a", "b", "c", etc.

    Un arc est créé si la capacité est strictement positive.
    L'étiquette de chaque arc affiche la capacité et, le cas échéant, le coût.
    """
    n = reseau["n"]
    capacites = reseau["capacites"]
    couts = reseau["couts"]

    mapping = mapping_noeuds(n)

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(mapping[i])

    for i in range(n):
        for j in range(n):
            if capacites[i][j] > 0:
                etiquette = f"{capacites[i][j]}"
                if couts is not None:
                    etiquette += f", cout={couts[i][j]}"
                G.add_edge(mapping[i], mapping[j], label=etiquette, capacity=capacites[i][j])
    return G


def afficher_graphe_flot(G):
    """
    Affiche graphiquement le graphe du réseau de flot.

    Caractéristiques :
      - Positionnement des nœuds en cercle.
      - La source ("s") est affichée en vert et le puits ("t") en rouge, les autres en bleu clair.
      - Les arêtes sont tracées en ligne droite avec des flèches indiquant le sens.
      - Les labels des arêtes (affichant capacité et coût) sont affichés sur un fond blanc semi-transparent.
    """
    n = G.number_of_nodes()

    # Définir la couleur en fonction de l'étiquette
    couleurs = []
    for node in G.nodes():
        if node == "s":
            couleurs.append("green")
        elif node == "t":
            couleurs.append("red")
        else:
            couleurs.append("lightblue")

    pos = nx.circular_layout(G)

    plt.figure(figsize=(10, 6))
    nx.draw_networkx_nodes(G, pos, node_color=couleurs, node_size=2000, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="-|>", arrowsize=20,
                           connectionstyle="arc3,rad=0.0", edge_color="black")

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="blue", font_size=10,
                                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.7))

    plt.title("Représentation graphique du réseau de flot")
    plt.axis("off")
    plt.show(block=True)


def construire_matrice_flot(n):
    """
    :param n:
    :return: [[0, 0, 0, ...], [0, 0, 0, ...], [0, 0, 0, ...], [0, 0, 0, ...], ...]
    """
    return [[0] * n] * n


def maj_matrice_flot(flot, maj):
    """
    Addition des deux matrices flot et maj pour renvoyer une matrice de flot à jour.
    """
    new_flot = flot
    for i in range(len(new_flot)):
        for j in range(len(new_flot[i])):
            new_flot[i][j] += maj[i][j]
    return new_flot


def construire_reseau_residuel(capacites, flot):
    """
    Construction de la matrice de réseau résiduel.
    :param capacites:
    :param flot:
    :return: [[[1, 5], [2, 3], [0, 5], ...], [[8, 11], [0, 9], [4, 6], ...], [[1, 2], [0, 3], [5, 7], ...], ...]
    Le premier chiffre correspond à ce qui peut encore être envoyé depuis le sommet correspondant à la ligne vers le sommet correspondant à la colonne.
    Le deuxième correpond au flot déjà présent entre ces deux sommets et qu'il faudrait envoyer dans l'autre sens si l'on souhaitait annuler le débit.
    """
    reseau_residuel = []
    for i in range(len(capacites)):
        sommet = []
        for j in range(len(capacites[i])):
            sommet.append([capacites[i][j] - flot[i][j], flot[i][j]])
        reseau_residuel.append(sommet)
    return reseau_residuel


def maximiser_flot(reseau_residuel):
    maximise = False
    while not maximise:

