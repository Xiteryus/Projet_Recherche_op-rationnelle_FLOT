import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate
from random import randint

def lire_reseau_flot(filename):
    """
    Lit un fichier texte contenant la table d'un réseau de flot.

    Format attendu :
      - Première ligne : entier n (nombre de sommets).
      - Lignes 2 à n+1 : matrice n×n des capacités.
      - Lignes n+2 à 2n+1 (optionnel) : matrice n×n des coûts.
    """
    with open(filename, 'r') as f:
        lignes = [l.strip() for l in f if l.strip()]
    if len(lignes) < 2:
        raise ValueError(f"Fichier '{filename}' trop court.")
    n = int(lignes[0])
    if len(lignes) < 1 + n:
        raise ValueError(f"Il manque des lignes pour la matrice des capacités (n={n}).")
    # Capacités
    caps = []
    for i in range(1, 1+n):
        row = list(map(int, lignes[i].split()))
        if len(row) != n:
            raise ValueError(f"Ligne {i+1} : attendu {n} colonnes, trouvé {len(row)}.")
        caps.append(row)
    capacites = np.array(caps, dtype=int)
    # Coûts (optionnel)
    couts = None
    start = 1 + n
    if len(lignes) >= start + n:
        costs = []
        for i in range(start, start + n):
            row = list(map(int, lignes[i].split()))
            if len(row) != n:
                raise ValueError(f"Ligne {i+1} : attendu {n} colonnes pour les coûts.")
            costs.append(row)
        couts = np.array(costs, dtype=int)
    return {"n": n, "capacites": capacites, "couts": couts}


def mapping_noeuds(n):
    """
    index -> label : 0->'s', n-1->'t', 1..n-2 -> 'a','b',...
    """
    m = {0: "s", n-1: "t"}
    for i in range(1, n-1):
        m[i] = chr(ord('a') + i - 1)
    return m


def afficher_tableau(mat, titre="Tableau", fmt="grid"):
    """
    Affiche une matrice numpy square avec étiquettes s, a, b, ..., t.
    """
    n = mat.shape[0]
    labels = [mapping_noeuds(n)[i] for i in range(n)]
    table = [[str(mat[i][j]) if mat[i][j] != float('inf') else "∞"
              for j in range(n)] for i in range(n)]
    print(titre)
    print(tabulate(table, headers=labels, showindex=labels, tablefmt=fmt))


def afficher_tableau_flot(flux, capacites):
    """
    Affiche le flot sous la forme 'f/c' pour chaque arête.
    """
    n = flux.shape[0]
    labels = [mapping_noeuds(n)[i] for i in range(n)]
    table = []
    for i in range(n):
        row = []
        for j in range(n):
            cap = capacites[i][j]
            f   = flux[i][j]
            if cap == 0:
                row.append("0")
            else:
                row.append(f"{f}/{cap}")
        table.append(row)
    print(tabulate(table, headers=labels, showindex=labels, tablefmt="grid"))


def construire_graphe_flot(reseau):
    """
    Retourne un DiGraph où chaque arête a attributes:
      - capacity
      - cost (None si absent)
      - label "(cap ; cost)" ou "cap"
    """
    n     = reseau["n"]
    caps  = reseau["capacites"]
    costs = reseau["couts"]
    mapping = mapping_noeuds(n)

    G = nx.DiGraph()
    for i in range(n):
        G.add_node(mapping[i])
    for i in range(n):
        for j in range(n):
            cap = caps[i][j]
            if cap > 0:
                cost = costs[i][j] if costs is not None else None
                label = f"({cap} ; {cost})" if cost is not None else f"{cap}"
                G.add_edge(mapping[i], mapping[j],
                           capacity=cap, cost=cost, label=label)
    return G


def afficher_graphe_flot(G, flow_matrix=None, mode=None):
    """
    Affichage circulaire, nœuds colorés, arêtes droites, labels sur fond blanc.
    Si flow_matrix+mode donné, colore en rouge (max) ou bleu (min) les arêtes avec flux>0.
    """
    n = G.number_of_nodes()
    idx2lbl = mapping_noeuds(n)
    lbl2idx = {lbl: idx for idx,lbl in idx2lbl.items()}

    # couleurs nœuds
    node_colors = []
    for u in G.nodes():
        if u == "s":   node_colors.append("green")
        elif u == "t": node_colors.append("red")
        else:          node_colors.append("lightblue")

    pos = nx.circular_layout(G)
    plt.figure(figsize=(10,6))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=2000, edgecolors="black")
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")

    # couleurs arêtes
    edge_colors = []
    for u,v in G.edges():
        if flow_matrix is not None and mode in ("max","min"):
            i,j = lbl2idx[u], lbl2idx[v]
            edge_colors.append("red" if mode=="max" and flow_matrix[i][j]>0
                               else "blue" if mode=="min" and flow_matrix[i][j]>0
                               else "black")
        else:
            edge_colors.append("black")

    nx.draw_networkx_edges(
        G, pos,
        arrows=True, arrowstyle='-|>', arrowsize=20,
        connectionstyle='arc3,rad=0.0',
        edge_color=edge_colors
    )

    # labels arêtes
    edge_labels = {(u,v): data["label"] for u,v,data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color="blue", font_size=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7)
    )

    title = "Représentation graphique du réseau de flot"
    if mode=="max":
        title += " — Flot maximal (rouge)"
    elif mode=="min":
        title += " — Flot minimal (bleu)"
    plt.title(title)
    plt.axis("off")
    plt.show(block=True)

#-------------------------------------------------------------------------------------------------------------------------------
#FLOT MAX


def ford_fulkerson(reseau):
    """
    Edmonds–Karp + affichage tabulate du flot f/c.
    Renvoie (max_flow, flow_matrix).
    """
    n = reseau["n"]
    caps = reseau["capacites"]
    residual = caps.copy()
    flow = np.zeros((n,n), dtype=int)
    s, t = 0, n-1

    def bfs(parent):
        visited = [False]*n
        q = [s]; visited[s]=True
        parent[:] = [-1]*n
        while q:
            u = q.pop(0)
            for v in range(n):
                if not visited[v] and residual[u][v]>0:
                    visited[v]=True; parent[v]=u; q.append(v)
                    if v==t: return True
        return False

    parent = [-1]*n
    max_flow = 0
    while bfs(parent):
        cf = float('inf'); v=t
        while v!=s:
            u=parent[v]; cf=min(cf, residual[u][v]); v=u
        max_flow += cf
        v=t
        while v!=s:
            u=parent[v]
            flow[u][v]+=cf; flow[v][u]-=cf
            residual[u][v]-=cf; residual[v][u]+=cf
            v=u

    print(f"\nFlot maximal = {max_flow}\n")
    afficher_tableau_flot(flow, caps)
    return max_flow, flow


def Iteration_ford_fulkerson(reseau):
    """
    Version affichant les Iterations du Ford-Fulkerson & (Edmonds-Karp)
    Renvoie (max_flow, flow_matrix).
    """
    n = reseau["n"]
    caps = reseau["capacites"]
    residual = caps.copy()
    s, t = 0, n - 1
    flow = np.zeros((n, n), dtype=int)
    iteration = 1
    labels = mapping_noeuds(n)

    def bfs_trace(parent):
        visited = [False] * n
        q = [s]
        visited[s] = True
        parent[:] = [-1] * n
        parcours = [labels[s]]
        while q:
            u = q.pop(0)
            for v in range(n):
                if not visited[v] and residual[u][v] > 0:
                    visited[v] = True
                    parent[v] = u
                    q.append(v)
                    parcours.append(labels[v])
                    if v == t:
                        return True, parcours
        return False, parcours
    parent = [-1] * n
    max_flow = 0

    while True:
        found, parcours = bfs_trace(parent)
        if not found:
            break

        cf = float("inf")
        v = t
        path = []
        while v != s:
            u = parent[v]
            cf = min(cf, residual[u][v])
            path.append(v)
            v = u
        path.append(s)
        path = path[::-1]

        print(f"\nItération : {iteration} :")
        print("Parcours en largeur :")
        for i in range(n):
            if parent[i] != -1 and i != s:
                print(f"Π({labels[i]}) = {labels[parent[i]]}; ", end="")
        print()
        print(f"Chaine améliorante : {''.join([labels[i] for i in path])} de flot {cf}.")

        v = t
        while v != s:
            u = parent[v]
            flow[u][v] += cf
            flow[v][u] -= cf
            residual[u][v] -= cf
            residual[v][u] += cf
            v = u

        print("Modifications sur le graphe résiduel :")
        afficher_tableau_flot(flow, caps)

        max_flow += cf
        iteration += 1

    print(f"\n Flot maximal = {max_flow}")
    return max_flow, flow

def push_relabel(reseau):
    """
    Push–Relabel + affichage tabulate du flot.
    Renvoie (max_flow, flow_matrix).
    """
    n = reseau["n"]
    caps = reseau["capacites"]
    residual = caps.copy()
    flow = np.zeros((n,n), dtype=int)
    s, t = 0, n-1
    height = [0]*n; excess=[0]*n

    height[s]=n
    for v in range(n):
        if caps[s][v]>0:
            flow[s][v]=caps[s][v]; flow[v][s]=-caps[s][v]
            excess[v]=caps[s][v]; excess[s]-=caps[s][v]
            residual[s][v]=0; residual[v][s]=caps[s][v]

    def push(u,v):
        delta=min(excess[u], residual[u][v])
        flow[u][v]+=delta; flow[v][u]-=delta
        residual[u][v]-=delta; residual[v][u]+=delta
        excess[u]-=delta; excess[v]+=delta

    def relabel(u):
        height[u]=1+min(height[v] for v in range(n) if residual[u][v]>0)

    active=[u for u in range(n) if u not in (s,t) and excess[u]>0]
    while active:
        u=active[0]; pushed=False
        for v in range(n):
            if residual[u][v]>0 and height[u]==height[v]+1:
                push(u,v); pushed=True
                if v not in (s,t) and excess[v]>0 and v not in active:
                    active.append(v)
                if excess[u]==0: break
        if not pushed: relabel(u)
        if excess[u]==0: active.pop(0)

    mf = sum(flow[s])
    print(f"\nFlot maximal = {mf}\n")
    afficher_tableau_flot(flow, caps)
    return mf, flow

#-------------------------------------------------------------------------------------------------------------------------------
#FLOT MIN


def min_cost_flow(reseau, target):
    """
    Min cost flow via Bellman-Ford + affichage tabulate.
    Renvoie (flow_value, cost_total, flow_matrix).
    """
    n = reseau["n"]
    caps = reseau["capacites"]
    costs = reseau["couts"]
    if costs is None:
        raise ValueError("Pas de matrice de coûts")
    residual = caps.copy()
    flow = np.zeros((n,n), dtype=int)
    s,t = 0,n-1
    total_flow=0; total_cost=0

    while total_flow<target:
        dist=[float('inf')]*n; parent=[-1]*n; dist[s]=0
        for _ in range(n-1):
            upd=False
            for u in range(n):
                for v in range(n):
                    if residual[u][v]>0 and dist[u]+costs[u][v]<dist[v]:
                        dist[v]=dist[u]+costs[u][v]; parent[v]=u; upd=True
            if not upd: break
        if dist[t]==float('inf'): break

        inc = target-total_flow
        v=t
        while v!=s:
            u=parent[v]; inc=min(inc, residual[u][v]); v=u
        total_flow+=inc; total_cost+=inc*dist[t]
        v=t
        while v!=s:
            u=parent[v]
            flow[u][v]+=inc; flow[v][u]-=inc
            residual[u][v]-=inc; residual[v][u]+=inc
            v=u

    print(f"\nFlot obtenu = {total_flow}   Coût total = {total_cost}\n")
    afficher_tableau_flot(flow, caps)
    return total_flow, total_cost, flow

def Iteration_min_cost_flow(reseau, target):
    """
    Version affichant les Iterations du Min Cost Flow & Affichage tabulate.
    Renvoie (flow_value, cost_total, flow_matrix).
    """
    n = reseau["n"]
    caps = reseau["capacites"]
    costs = reseau["couts"]
    if costs is None:
        raise ValueError("Pas de matrice de coûts")
    residual = caps.copy()
    flow = np.zeros((n,n), dtype=int)
    s, t = 0, n-1
    total_flow = 0 ;total_cost = 0
    iteration = 1
    labels = [mapping_noeuds(n)[i] for i in range(n)]

    while total_flow < target:
        print(f"\nItération {iteration} :")
        dist = [float('inf')] * n
        parent = [-1] * n
        dist[s] = 0


        history = [["0" if i == s else "∞" for i in range(n)]]

        for step in range(1, n):
            updated = False
            new_dist = dist[:]
            for u in range(n):
                for v in range(n):
                    if residual[u][v] > 0 and dist[u] + costs[u][v] < new_dist[v]:
                        new_dist[v] = dist[u] + costs[u][v]
                        parent[v] = u
                        updated = True
            dist = new_dist[:]
            history.append([str(d) if d < float('inf') else "∞" for d in dist])
            if not updated:
                break
        headers = ["k"] + reseau.get("noms", [f"s{i}" for i in range(n)])
        rows = [[i] + history[i] for i in range(len(history))]
        print("\nTable de Bellman-Ford :")
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        if dist[t] == float('inf'):
            print("Plus de chaîne améliorante trouvée.")
            break
        print("\n")
        # Retrouver et afficher le chemin améliorant
        chemin = []
        v = t
        while v != s:
            chemin.append(v)
            v = parent[v]
        chemin.append(s)
        chemin.reverse()
        inc = target - total_flow
        v = t
        while v != s:
            u = parent[v]
            inc = min(inc, residual[u][v])
            v = u
        print("Chaîne améliorante :", ' → '.join(mapping_noeuds(n)[i] for i in chemin))
        print(f"Flot de cette iteration sur cette chaîne : {inc}")
        total_flow += inc
        total_cost += inc * dist[t]
        if total_flow < target:
            print(f"Le flot total est : {total_flow} < {target} , donc on continue")
        print("Modifications sur le graphe résiduel :")
        v = t
        while v != s:
            u = parent[v]
            flow[u][v] += inc
            flow[v][u] -= inc
            residual[u][v] -= inc
            residual[v][u] += inc
            v = u

        table = []
        for i in range(n):
            row = []
            for j in range(n):
                row.append(str(residual[i][j]))
            table.append(row)
        print(tabulate(table, headers=labels, showindex=labels, tablefmt="grid"))
        iteration += 1

    print(f"\n Flot obtenu = {total_flow}   Coût total = {total_cost}")
    afficher_tableau_flot(flow, caps)
    return total_flow, total_cost, flow


#-------------------------------------------------------------------------------------------------------------------------------
#COMPLEXITE


def generer_cout_aleatoire(n):
    cout = []
    for i in range(n):
        sommet = []
        for j in range(n):
            sommet.append(randint(0, 10))
        cout.append(sommet)
    return cout


def generer_flot_aleatoire(n):
    flot = []
    for i in range(n):
        sommet = []
        for j in range(n):
            sommet.append(randint(0, 200))
        flot.append(sommet)
    return flot


def creer_reseau_flot(n, choixCout):
    capacites = generer_flot_aleatoire(n)
    capacites = np.array(capacites)
    if choixCout == "Y" or choixCout == "y":
        couts = generer_cout_aleatoire(n)
        couts = np.array(couts)
    else:
        couts = None
    return {"n": n, "capacites": capacites, "couts": couts}

def choix_fichier():
    reseau = None
    randomness = None
    # Boucle de choix d'un fichier existant ou random
    while randomness is None:
        randomness = input("Choix d'un fichier déjà existant [Y/N] : ")
        if randomness == 'Y' or randomness == 'y':
            # Lecture du fichier jusqu'à succès
            while reseau is None:
                filename = input("Entrez le nom du fichier de la table du réseau de flot (ex. : reseau.txt) : ")
                try:
                    reseau = lire_reseau_flot(filename)
                    print(f"Fichier '{filename}' chargé avec succès.")
                except Exception as e:
                    print("Erreur lors de la lecture du fichier :", e)
                    reseau = None
        elif randomness == 'N' or randomness == 'n':
            nbSommets = int(input("Choisissez le nombre de sommets : "))
            choixCout = input("Associer des coûts à chaque arrête ? [Y/N] : ")
            if choixCout not in ["Y", "y", "N", "n"]:
                print("Entrez 'Y' ou 'N' !")
                reseau = None
            else:
                reseau = creer_reseau_flot(nbSommets, choixCout)
        else:
            print("Entrez 'Y' ou 'N' !")
            randomness = None
    return reseau