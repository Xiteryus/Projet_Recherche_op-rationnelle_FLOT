from functions import *


def main():
    reseau = None
    # Boucle tant que la lecture du fichier échoue
    while reseau is None:
        filename = input("Entrez le nom du fichier de la table du réseau de flot (ex. : reseau.txt) : ")
        try:
            reseau = lire_reseau_flot(filename)
            print(f"Fichier '{filename}' chargé avec succès.")
        except Exception as e:
            print("Erreur lors de la lecture du fichier :", e)
            reseau = None

    choix = 0
    while choix != 5:
        print("\nMenu :")
        print("1. Choisir un autre fichier de réseau de flot")
        print("2. Afficher la matrice des capacités (avec étiquettes de nœuds)")
        print("3. Afficher la matrice des coûts (si présente, avec étiquettes de nœuds)")
        print("4. Afficher le graphe du réseau de flot")
        print("5. Quitter")
        saisie = input("Entrez votre choix (1-5) : ")

        if not saisie.isdigit():
            print("Veuillez entrer un nombre valide.")
            continue

        choix = int(saisie)

        if choix == 1:
            filename = input("Entrez le nom du fichier de la table du réseau de flot (ex. : reseau.txt) : ")
            try:
                reseau = lire_reseau_flot(filename)
                print(f"Fichier '{filename}' chargé avec succès.")
            except Exception as e:
                print("Erreur lors de la lecture du fichier :", e)
        elif choix == 2:
            print("\nMatrice des capacités :")
            afficher_tableau(reseau["capacites"], titre="Matrice des capacités")
        elif choix == 3:
            if reseau["couts"] is not None:
                print("\nMatrice des coûts :")
                afficher_tableau(reseau["couts"], titre="Matrice des coûts")
            else:
                print("Aucune matrice des coûts n'a été trouvée dans le fichier.")
        elif choix == 4:
            G = construire_graphe_flot(reseau)
            afficher_graphe_flot(G)
        elif choix == 5:
            print("Merci d'avoir utilisé le programme.")
        else:
            print("Veuillez entrer un nombre entre 1 et 5.")


if __name__ == "__main__":
    main()
