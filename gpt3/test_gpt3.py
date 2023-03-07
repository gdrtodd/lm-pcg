from Levenshtein import distance
import networkx as nx
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, get_context

from sokoban_solvers import EnhancedAStarAgent, State

microban_data = pd.read_hdf("./caches/microban_data.h5", key="data")
microban_flips_data = pd.read_hdf("./caches/microban_flips_data.h5", key="data")
microban_flips_rotations_data = pd.read_hdf("./caches/microban_flips_rotations_data.h5", key="data")

microban_samples = pd.read_csv("./caches/microban_orig_sample_exp.csv")
microban_flips_samples = pd.read_csv("./caches/microban_flips_sample_exp.csv")
microban_flips_rotations_samples = pd.read_csv("./caches/microban_flips_rotation_sample_exp.csv")

def get_novelty(samples, data, threshold=5, verbose=False):
    novelties, nearest_levels = [], []

    for sample in samples["level"]:
        dists = [distance(sample, level) for level in data["level"]]
        min_dist, nearest_level = min(dists), list(data["level"])[dists.index(min(dists))]

        novelties.append(min_dist > threshold)
        nearest_levels.append(nearest_level)

        if verbose:
            line_offset = len(sample.split("\n")[0]) - len("SAMPLE") # just for lining things up
            print(f"\nSAMPLE{' ' * line_offset}\t\t \t\tNEAREST LEVEL")
            for l1, l2 in zip(sample.split("\n"), nearest_level.split("\n")):
                print(f"{l1.replace('-', ' ')}\t\t|\t\t{l2.replace('-', ' ')}")
            print(f"Distance: {min_dist}")

    return novelties, nearest_levels

def get_solvability(samples):
    solvabilities = []

    with Pool(32) as pool:
        solvabilities = list(tqdm(pool.imap(get_solvability_single, samples["level"]), total=len(samples), desc="Computing sovability"))

    return solvabilities

def get_solvability_single(sample):
    lines = [line for line in sample.split("\n") if line]
    solver = EnhancedAStarAgent()
    level_state = State().stringInitialize(lines)
    max_iter = 10000000

    try:
        solution, node, iters = solver.getSolution(level_state, maxIterations=max_iter)
        return node.checkWin()
    except (IndexError, TypeError) as err:
        return False

def get_diversity(samples, clique_limit=1000000):
    '''
    Returns the 'diversity' of a set of levels, defined as the size of the largest subset of levels
    that are all at least 'self.novelty_threshold' edit distance away from each other. We compute this 
    by constructing a graph where levels are adjacent if their edit distance is at least the threshold,
    and then finding the size of the largest clique in the graph.
    '''

    levels = list(samples["level"])

    graph = nx.Graph()
    graph.add_nodes_from(range(len(levels)))

    edges = []
    for i in range(len(levels)):
        for j in range(i+1, len(levels)):
            if distance(levels[i], levels[j]) >= 5:
                edges.append((i, j))

    graph.add_edges_from(edges)

    biggest_clique = 1
    num_cliques = 0

    for clique in nx.find_cliques(graph):
        if len(clique) > biggest_clique:
            biggest_clique = len(clique)
        num_cliques += 1

        if num_cliques > clique_limit:
            break


    return biggest_clique

print("\nMicroban")
novelties, nearest_levels = get_novelty(microban_samples, microban_data, verbose=False)
solvabiltiies = get_solvability(microban_samples)
diversity = get_diversity(microban_samples)


combined = [novel and solvable for novel, solvable in zip(novelties, solvabiltiies)]
novel_playable_levels = microban_samples.loc[combined]
score = get_diversity(novel_playable_levels) / len(microban_samples)

print(f"-Prop novel: {sum(novelties) / len(microban_samples)}")
print(f"-Prop solvable: {sum(solvabiltiies) / len(microban_samples)}")
print(f"-Diversity: {diversity / len(microban_samples)}")
print(f"-Prop novel and playable: {sum(combined) / len(microban_samples)}")
print(f"-Score: {score}")

print("\nMicroban Flips")
novelties, nearest_levels = get_novelty(microban_flips_samples, microban_flips_data, verbose=False)
solvabiltiies = get_solvability(microban_flips_samples)
diversity = get_diversity(microban_flips_samples)

combined = [novel and solvable for novel, solvable in zip(novelties, solvabiltiies)]
novel_playable_levels = microban_flips_samples.loc[combined]
score = get_diversity(novel_playable_levels) / len(microban_flips_samples)

print(f"-Prop novel: {sum(novelties) / len(microban_flips_samples)}")
print(f"-Prop solvable: {sum(solvabiltiies) / len(microban_flips_samples)}")
print(f"-Diversity: {diversity / len(microban_flips_samples)}")
print(f"-Prop novel and playable: {sum(combined) / len(microban_flips_samples)}")
print(f"-Score: {score}")

print("\nMicroban Flips Rotations")
novelties, nearest_levels = get_novelty(microban_flips_rotations_samples, microban_flips_rotations_data, verbose=False)
solvabiltiies = get_solvability(microban_flips_rotations_samples)
diversity = get_diversity(microban_flips_rotations_samples)

combined = [novel and solvable for novel, solvable in zip(novelties, solvabiltiies)]
novel_playable_levels = microban_flips_rotations_samples.loc[combined]
score = get_diversity(novel_playable_levels) / len(microban_flips_rotations_samples)

print(f"-Prop novel: {sum(novelties) / len(microban_flips_rotations_samples)}")
print(f"-Prop solvable: {sum(solvabiltiies) / len(microban_flips_rotations_samples)}")
print(f"-Diversity: {diversity / len(microban_flips_rotations_samples)}")
print(f"-Prop novel and playable: {sum(combined) / len(microban_flips_rotations_samples)}")
print(f"-Score: {score}")