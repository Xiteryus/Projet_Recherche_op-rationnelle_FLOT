from functions import *

def main():
    #reseau = choix_fichier()
    choix = 0
    while choix != 8:
        print("\nMenu :")
        print("1. Choisir un autre fichier de réseau de flot")
        print("2. Afficher la matrice des capacités")
        print("3. Afficher la matrice des coûts")
        print("4. Afficher le graphe du réseau de flot (sans flot)")
        print("5. Calculer le flot maximal (Ford–Fulkerson) et afficher")
        print("6. Calculer le flot maximal (Push–Relabel) et afficher")
        print("7. Calculer le flot à coût minimal et afficher")
        print("8. Quitter")
        saisie = input("Entrez votre choix (1-8) : ")

        if not saisie.isdigit():
            print("Veuillez entrer un nombre valide.")
            continue
        choix = int(saisie)

        if choix == 1:
            reseau = choix_fichier()
        elif choix == 2:
            # On peut réutiliser afficher_tableau si nécessaire :
            from functions import afficher_tableau
            print("\nMatrice des capacités :")
            afficher_tableau(reseau["capacites"], titre="Capacités")

        elif choix == 3:
            from functions import afficher_tableau
            if reseau["couts"] is not None:
                print("\nMatrice des coûts :")
                afficher_tableau(reseau["couts"], titre="Coûts")
            else:
                print("Aucune matrice des coûts trouvée.")

        elif choix == 4:
            G = construire_graphe_flot(reseau)
            afficher_graphe_flot(G)  # mode None

        elif choix == 5:
            # Ford–Fulkerson
            #choix d'affichage des itérations
            print("Souhaitez-vous afficher les itérations du Ford–Fulkerson ?")
            print("1. Oui (visualiser les étapes)")
            print("2. Non (juste le résultat final)")
            choixbis = input("Entrez votre choix (1 ou 2) : ")
            if choixbis == "1":
                max_flow, flow = Iteration_ford_fulkerson(reseau)
            else:
                max_flow, flow = ford_fulkerson(reseau)
            G = construire_graphe_flot(reseau)
            # on passe flow et mode='max' pour colorier en rouge
            afficher_graphe_flot(G, flow_matrix=flow, mode='max')

        elif choix == 6:
            # Push–Relabel
            max_flow, flow = push_relabel(reseau)
            G = construire_graphe_flot(reseau)
            afficher_graphe_flot(G, flow_matrix=flow, mode='max')

        elif choix == 7:
            if reseau["couts"] is None:
                print("Impossible : pas de matrice des coûts disponible.")
                continue
            try:
                target = int(input("Entrez la valeur de flot souhaitée : "))
            except ValueError:
                print("Valeur invalide.")
                continue
            print("Souhaitez-vous afficher les itérations du flot à coût minimal ?")
            print("1. Oui ")
            print("2. Non ")
            choixbis = input("Entrez votre choix (1 ou 2) : ")

            if choixbis == "1":
                total_flow, total_cost, flow = Iteration_min_cost_flow(reseau, target)
            else:
                total_flow, total_cost, flow = min_cost_flow(reseau, target)
            G = construire_graphe_flot(reseau)
            # mode='min' pour colorier en bleu
            afficher_graphe_flot(G, flow_matrix=flow, mode='min')

        elif choix == 8:
            print("Merci d'avoir utilisé le programme.")

        else:
            print("Veuillez entrer un nombre entre 1 et 8.")

if __name__ == "__main__":
    main()
