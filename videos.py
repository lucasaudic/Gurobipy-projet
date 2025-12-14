import sys
import argparse
import gurobipy as gp
from gurobipy import GRB

def solve(input_file):
    print(f"Reading input from {input_file}...")
    with open(input_file, 'r') as f:
        V, E, R, C, X = map(int, f.readline().split())
        
        video_sizes = list(map(int, f.readline().split()))
        
        endpoints = []
        for _ in range(E):
            L_dc, K = map(int, f.readline().split())
            cache_connections = {}
            for _ in range(K):
                c, L_c = map(int, f.readline().split())
                cache_connections[c] = L_c
            endpoints.append({'L_dc': L_dc, 'caches': cache_connections})
            
        requests = []
        for _ in range(R):
            v, e, n = map(int, f.readline().split())
            requests.append({'v': v, 'e': e, 'n': n})

    print("Building model...")
    m = gp.Model("videos")
    
    x = {} 
    
    relevant_pairs = set()
    for req in requests:
        v = req['v']
        e = req['e']
        for c in endpoints[e]['caches']:
            relevant_pairs.add((c, v))

    for c, v in relevant_pairs:
        x[c, v] = m.addVar(vtype=GRB.BINARY, name=f"x_{c}_{v}")

    for c in range(C):
        m.addConstr(
            gp.quicksum(video_sizes[v] * x[c, v] for v in range(V) if (c, v) in x) <= X,
            name=f"capacity_cache_{c}"
        )

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
            
        sorted_caches = sorted(caches.items(), key=lambda item: item[1])
        
        L_dc = endpoints[e]['L_dc']
        valid_caches = [(c, lat) for c, lat in sorted_caches if lat < L_dc]
        
        if not valid_caches:
            continue

        values = [L_dc - lat for c, lat in valid_caches]
        
        for i in range(len(valid_caches)):
            if i < len(valid_caches) - 1:
                marginal_gain = values[i] - values[i+1]
            else:
                marginal_gain = values[i]
            if marginal_gain == 0:
                continue
                
            z = m.addVar(vtype=GRB.BINARY, obj=marginal_gain * n, name=f"z_{v}_{e}_{i}")
            
            subset_caches = [c for c, _ in valid_caches[0:i+1]]
            m.addConstr(z <= gp.quicksum(x[c, v] for c in subset_caches), name=f"cover_{v}_{e}_{i}")

    m.ModelSense = GRB.MAXIMIZE
    m.Params.MipGap = 0.005 
    m.Params.TimeLimit = 300 

    print("Exporting model to videos.mps...")
    m.write("videos.mps")

    print("Optimizing...")
    m.optimize()

    print("Generating output...")
    
    solution = {} 
    if m.SolCount > 0:
        for (c, v), var in x.items():
            if var.X > 0.5: 
                if c not in solution:
                    solution[c] = []
                solution[c].append(v)
    
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
