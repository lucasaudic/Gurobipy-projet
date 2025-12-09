import sys
import argparse
import gurobipy as gp
from gurobipy import GRB

def solve(input_file):
    print(f"Reading input from {input_file}...")
    with open(input_file, 'r') as f:
        # Lire la première ligne : V (vidéos), E (endpoints), R (requêtes), C (serveurs de cache), X (capacité)
        V, E, R, C, X = map(int, f.readline().split())
        
        # Deuxième ligne : taille de chaque vidéo
        video_sizes = list(map(int, f.readline().split()))
        
        endpoints = []
        for _ in range(E):
            # Pour chaque endpoint : Latence vers le datacenter (L_dc) et nombre de caches connectés (K)
            L_dc, K = map(int, f.readline().split())
            cache_connections = {}
            for _ in range(K):
                # Connexion : ID du cache (c) et latence vers ce cache (L_c)
                c, L_c = map(int, f.readline().split())
                cache_connections[c] = L_c
            endpoints.append({'L_dc': L_dc, 'caches': cache_connections})
            
        requests = []
        for _ in range(R):
            # Pour chaque requête : ID de la vidéo (v), ID de l'endpoint (e), nombre de demandes (n)
            v, e, n = map(int, f.readline().split())
            requests.append({'v': v, 'e': e, 'n': n})

    print("Building model...")
    m = gp.Model("videos")
    
    # -------------------------------------------------------------
    # 1. Variables de décision
    # -------------------------------------------------------------
    
    # x[c, v] = 1 si la vidéo v est stockée dans le cache c, 0 sinon
    x = {} 
    
    # Optimisation : On ne crée des variables que pour les couples (cache, vidéo) pertinents
    # c'est-à-dire ceux qui peuvent vraiment servir une requête existante.
    relevant_pairs = set()
    for req in requests:
        v = req['v']
        e = req['e']
        for c in endpoints[e]['caches']:
            relevant_pairs.add((c, v))

    for c, v in relevant_pairs:
        x[c, v] = m.addVar(vtype=GRB.BINARY, name=f"x_{c}_{v}")

    # -------------------------------------------------------------
    # 2. Contraintes
    # -------------------------------------------------------------
    
    # Contrainte de capacité pour chaque serveur de cache :
    # La somme des tailles des vidéos stockées ne doit pas dépasser X.
    for c in range(C):
        m.addConstr(
            gp.quicksum(video_sizes[v] * x[c, v] for v in range(V) if (c, v) in x) <= X,
            name=f"capacity_cache_{c}"
        )

    # -------------------------------------------------------------
    # 3. Fonction Objectif : Maximiser les économies de latence
    # -------------------------------------------------------------
    
    # L'objectif est de maximiser le gain total (latence gagnée par rapport au datacenter).
    # Le gain pour une requête dépend du MEILLEUR cache disponible.
    # C'est une fonction "max" qui est difficile à modéliser directement.
    # On utilise une astuce avec les gains marginaux.
    
    # D'abord, on regroupe les requêtes identiques (même vidéo v, même endpoint e)
    requests_summary = {}
    for req in requests:
        pair = (req['v'], req['e'])
        requests_summary[pair] = requests_summary.get(pair, 0) + req['n']
        
    print(f"Total requests: {len(requests)}")
    print(f"Unique request pairs (v, e): {len(requests_summary)}")

    total_requests_count = 0 
    
    for (v, e), n in requests_summary.items():
        total_requests_count += n
        caches = endpoints[e]['caches']
        
        if not caches:
            continue
            
        # On trie les caches connectés par latence croissante (du meilleur au moins bon)
        sorted_caches = sorted(caches.items(), key=lambda item: item[1])
        
        # On ne garde que ceux qui sont meilleurs que le datacenter
        L_dc = endpoints[e]['L_dc']
        valid_caches = [(c, lat) for c, lat in sorted_caches if lat < L_dc]
        
        if not valid_caches:
            continue

        # Calcul des gains potentiels : 
        # values[i] = gain si servi par le i-ème meilleur cache
        values = [L_dc - lat for c, lat in valid_caches]
        
        # On décompose le gain total en gains marginaux incrémentaux.
        # Si la vidéo est dans le meilleur cache (0), on gagne tout.
        # Sinon, si elle est dans le 2ème meilleur (1), on gagne moins, etc.
        # On introduit des variables auxiliaires z_{v,e,i}
        
        for i in range(len(valid_caches)):
            # Gain marginal apporté par le cache i par rapport au cache i+1
            if i < len(valid_caches) - 1:
                marginal_gain = values[i] - values[i+1]
            else:
                marginal_gain = values[i]
            
            if marginal_gain <= 0:
                continue
                
            # Variable auxiliaire z : vaut 1 si la vidéo est présente dans au moins un des caches du sous-ensemble {0, ..., i}
            z = m.addVar(vtype=GRB.BINARY, obj=marginal_gain * n, name=f"z_{v}_{e}_{i}")
            
            # Contrainte liant z aux variables x :
            # z <= somme(x pour les caches du sous-ensemble)
            # Puisque l'objectif maximise z, le solveur mettra z à 1 dès qu'une vidéo est dispo.
            subset_caches = [c for c, _ in valid_caches[0:i+1]]
            m.addConstr(z <= gp.quicksum(x[c, v] for c in subset_caches), name=f"cover_{v}_{e}_{i}")

    # Configuration du solveur
    m.ModelSense = GRB.MAXIMIZE
    m.Params.MipGap = 0.005 # On vise un gap de 0.5% ou moins
    m.Params.TimeLimit = 300 # Limite de temps raisonnable (5 min)

    print("Exporting model to videos.mps...")
    m.write("videos.mps")

    print("Optimizing...")
    m.optimize()

    print("Generating output...")
    
    # Récupération de la solution
    solution = {} # cache_id -> liste des vidéos
    if m.SolCount > 0:
        for (c, v), var in x.items():
            if var.X > 0.5: # Si la variable est à 1 (avec tolérance flottante)
                if c not in solution:
                    solution[c] = []
                solution[c].append(v)
    
    # Écriture du fichier de sortie
    with open("videos.out", "w") as f:
        f.write(f"{len(solution)}\n")
        for c, videos in solution.items():
            f.write(f"{c} {' '.join(map(str, videos))}\n")
    
    print("Done.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python videos.py <input_file>")
        sys.exit(1)
        
    solve(sys.argv[1])
