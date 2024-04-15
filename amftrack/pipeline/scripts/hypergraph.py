# If launched as a script, change the params in if __name__ == "__main__":
import os
import pickle
from itertools import combinations

import networkx as nx
import numpy as np
from keras.models import load_model
from scipy import sparse
from tqdm import tqdm

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Edge,
    Experiment,
    Node,
)
from amftrack.pipeline.functions.image_processing.experiment_util import get_all_edges
from amftrack.pipeline.functions.post_processing.util import is_in_ROI_node
from amftrack.util.sys import get_current_folders, update_plate_info


############################################ FUNCTIONS #############################################
def closest_point(point, points):
    dist_square = np.sum((points - point) ** 2, axis=1)
    min_index = np.argmin(dist_square)
    return points[min_index], dist_square[min_index]


def score_ml_2(exp, v1, v2, v, positions, model):
    intersection = positions[v]
    data = []
    for vi in [v1, v2]:
        edge = Edge(Node(v, exp), Node(vi, exp), exp)
        pixels = np.array(edge.pixel_list(last_index)) - intersection
        if np.linalg.norm(pixels[0]) > np.linalg.norm(pixels[-1]):
            pixels = np.flip(pixels, 0)
        pixels = (list(pixels) + [pixels[-1]] * length_pixel_list)[:length_pixel_list]
        data.extend(pixels)
    data = np.array(data).reshape(1, -1)
    score = (
        model.predict(data, verbose=0)[0]
        + model.predict(np.roll(data, 2 * length_pixel_list), verbose=0)[0]
    )
    return score


def edge_matches_ml_2(
    exp, v, EExtract, E, positions, edges_time_interval, model
) -> list:
    if len(EExtract) < 2:
        return []

    times = []
    vs = []
    for e in EExtract:
        v_i = E[e][0] if E[e][1] == v else E[e][1]
        time_interval = edges_time_interval.get(f"{v_i},{v}")
        if time_interval is None:
            time, _ = edges_time_interval[f"{v},{v_i}"]
        else:
            time = time_interval[1]
        vs.append(v_i)
        times.append(time)

    min_time = min(times)
    min_time_edges = [index for index, time in enumerate(times) if time == min_time]

    if len(min_time_edges) == 2:
        e1 = EExtract[min_time_edges[0]]
        e2 = EExtract[min_time_edges[1]]
        return [(e1, e2, "time_resolved")] + edge_matches_ml_2(
            exp,
            v,
            [
                edge_index
                for edge_index in EExtract
                if (edge_index != e1 and edge_index != e2)
            ],
            E,
            positions,
            edges_time_interval,
            model,
        )

    if len(min_time_edges) > 2:
        pairs = combinations(min_time_edges, 2)
        best_score = 0
        for index1, index2 in pairs:
            score = score_ml_2(exp, vs[index1], vs[index2], v, positions, model)
            if score > best_score:
                best_score = score
                e1 = EExtract[index1]
                e2 = EExtract[index2]
        return [(e1, e2, "geometricaly_resolved")] + edge_matches_ml_2(
            exp,
            v,
            [
                edge_index
                for edge_index in EExtract
                if (edge_index != e1 and edge_index != e2)
            ],
            E,
            positions,
            edges_time_interval,
            model,
        )

    min_time_edge = min_time_edges[0]
    second_min_time = min([time for time in times if time != min_time])
    second_time_edges = [
        index for index, time in enumerate(times) if time == second_min_time
    ]

    if len(second_time_edges) == 1:
        e1 = EExtract[min_time_edge]
        e2 = EExtract[second_time_edges[0]]
        return [(e1, e2, "time_resolved")] + edge_matches_ml_2(
            exp,
            v,
            [
                edge_index
                for edge_index in EExtract
                if (edge_index != e1 and edge_index != e2)
            ],
            E,
            positions,
            edges_time_interval,
            model,
        )

    e1 = EExtract[min_time_edge]
    best_score = 0
    for index in second_time_edges:
        score = score_ml_2(exp, vs[index], vs[min_time_edge], v, positions, model)
        if score > best_score:
            best_score = score
            best_index = index
    e2 = EExtract[best_index]
    return [(e1, e2, "geometricaly_resolved")] + edge_matches_ml_2(
        exp,
        v,
        [
            edge_index
            for edge_index in EExtract
            if (edge_index != e1 and edge_index != e2)
        ],
        E,
        positions,
        edges_time_interval,
        model,
    )


def hypergraph_from_graph_ml_2(
    exp, G, positions, edges_time_interval, model, get_time_resolved_intersections=False
):
    V = list(G.nodes())
    E = list(G.edges())
    e = len(E)
    v = len(V)

    H = [0] * e
    Cor = [[0] * 10 for _ in range(e)]

    if get_time_resolved_intersections:
        time_resolved_intersections = []

    # STEP 1
    for i in tqdm(V, desc="Processing vertices"):
        EExtract = [edge_idx for edge_idx, edge in enumerate(E) if i in edge]
        matches = edge_matches_ml_2(
            exp, i, EExtract, E, positions, edges_time_interval, model
        )
        for match in matches:
            e1, e2 = match[0], match[1]
            Cor[e1][Cor[e1].index(0)] = e2
            Cor[e2][Cor[e2].index(0)] = e1
        if (
            get_time_resolved_intersections
            and matches
            and matches[0][2] == "time_resolved"
        ):
            time_resolved_intersections.append(
                (EExtract, matches[0][0], matches[0][1], positions[i])
            )

    # STEP 2
    CurrentMark = 1
    for i in tqdm(range(e), desc="Processing stack"):
        if H[i] == 0:
            stack = [i]
            visited = set()  # To keep track of edges that have been added to the stack
            while stack:
                current = stack.pop()
                H[current] = CurrentMark
                # Only add edges to the stack that haven't been assigned to a hyperedge and aren't already on the stack
                related_edges = [
                    cor
                    for cor in Cor[current]
                    if cor != 0 and H[cor] == 0 and cor not in visited
                ]
                stack.extend(related_edges)
                visited.update(related_edges)
            CurrentMark += 1
    H = {edge: H[i] for i, edge in enumerate(E)}
    if get_time_resolved_intersections:
        return H, time_resolved_intersections
    return H


############################################ MAIN #############################################
def main(
    directory_targ,
    saving_path,
    plates,
    first_index,
    last_index,
    threshold,
    amount_of_border_segment,
    length_pixel_list,
    model_path,
):
    # Load Networks
    update_plate_info(directory_targ, local=True, strong_constraint=False)
    all_folders = get_current_folders(directory_targ, local=True)

    folders = all_folders.loc[all_folders["unique_id"] == plates[0]]
    folders = folders.loc[folders["/Analysis/nx_graph_pruned_width.p"] == True]

    folders = folders.sort_values(by="datetime")

    exp = Experiment(directory_targ)
    exp.dimX_dimY = 0, 0

    try:
        exp.load(folders.iloc[first_index : last_index + 1].copy(), suffix="_width")
    except:
        pass

    exp.save_location = exp.folders.iloc[0]["total_path"]  # type: ignore

    # Size of the segment in pixels
    segments_length = 5

    # ROI
    final_graph = exp.nx_graph[last_index]
    node_not_in_ROI = []
    for node in final_graph:
        if not is_in_ROI_node(Node(node, exp), last_index):
            node_not_in_ROI.append(node)
    final_graph.remove_nodes_from(node_not_in_ROI)

    # Segmentation
    label = max(final_graph.nodes) + 1
    graph_segemented_final = nx.empty_graph()
    nodes_pos = {}
    edges_indexes = {}
    segments_index = {}
    segments_center_final = []

    for edge in final_graph.edges:
        e = Edge(Node(edge[0], exp), Node(edge[1], exp), exp)
        edges_indexes[f"{edge[0]},{edge[1]}"] = []
        pixels = e.pixel_list(last_index)
        length = len(pixels)
        if length < segments_length:
            graph_segemented_final.add_edge(edge[0], edge[1])
            segments_index[f"{edge[0]},{edge[1]}"] = len(segments_center_final)
            edges_indexes[f"{edge[0]},{edge[1]}"].append(len(segments_center_final))
            central_point = np.mean(np.array(pixels), axis=0)
            segments_center_final.append(central_point)
            nodes_pos[edge[0]] = pixels[0]
            nodes_pos[edge[1]] = pixels[-1]
            continue

        for i in range(0, length, segments_length):
            sub_list = pixels[i : i + segments_length]
            if i == 0:
                graph_segemented_final.add_edge(edge[0], label)
                segments_index[f"{edge[0]},{label}"] = len(segments_center_final)
                edges_indexes[f"{edge[0]},{edge[1]}"].append(len(segments_center_final))
                central_point = np.mean(np.array(sub_list), axis=0)
                segments_center_final.append(central_point)
                nodes_pos[edge[0]] = sub_list[0]
                nodes_pos[label] = sub_list[-1]
                label += 1
            elif i + segments_length >= length:
                graph_segemented_final.add_edge(label - 1, edge[1])
                segments_index[f"{label-1},{edge[1]}"] = len(segments_center_final)
                edges_indexes[f"{edge[0]},{edge[1]}"].append(len(segments_center_final))
                central_point = np.mean(np.array(sub_list), axis=0)
                segments_center_final.append(central_point)
                nodes_pos[edge[1]] = sub_list[-1]
            else:
                graph_segemented_final.add_edge(label - 1, label)
                segments_index[f"{label-1},{label}"] = len(segments_center_final)
                edges_indexes[f"{edge[0]},{edge[1]}"].append(len(segments_center_final))
                central_point = np.mean(np.array(sub_list), axis=0)
                segments_center_final.append(central_point)
                nodes_pos[label] = sub_list[-1]
                label += 1

    print(
        f"amount of nodes in graph_segemented_final: {graph_segemented_final.number_of_nodes()}"
    )
    print(
        f"amount of segments in graph_segemented_final: {graph_segemented_final.number_of_edges()}"
    )

    array_segments_center_final = np.array(segments_center_final)

    # Segment activation
    segments_centers = []
    segments_min_distances = []
    array_segments_center = array_segments_center_final.copy()
    for time in reversed(range(last_index + 1)):
        print(f"Process time {time}")
        rows = []
        cols = []
        previous_edges = get_all_edges(exp, time)
        for edge in previous_edges:
            p_list = edge.pixel_list(time)
            row, col = zip(*p_list)
            rows.extend(row)
            cols.extend(col)

        data = np.ones(len(rows))
        points_matrix = sparse.csr_matrix((data, (rows, cols)))

        centers_distance = []
        new_centers = array_segments_center.copy()
        for index, center in enumerate(array_segments_center):
            xc, yc = center
            xc, yc = int(xc), int(yc)

            min_x, max_x = max(0, xc - 4 * segments_length), xc + 4 * segments_length
            min_y, max_y = max(0, yc - 4 * segments_length), yc + 4 * segments_length
            coords = points_matrix[min_x:max_x, min_y:max_y].nonzero()
            coords = np.column_stack(coords)
            if not coords.shape[0]:
                centers_distance.append(32 * (segments_length**2))
                continue

            xc -= min_x
            yc -= min_y

            new_center, min_dist = closest_point([xc, yc], coords)
            centers_distance.append(min_dist)
            if min_dist < threshold:
                new_centers[index] = new_center + np.array([min_x, min_y])

        array_segments_center = new_centers
        segments_centers.append(new_centers)
        segments_min_distances.append(centers_distance)

    segments_min_distances.reverse()
    # Index t are the centers of the segments at time t
    segments_centers.reverse()

    segments_min_distances_array = np.array(segments_min_distances)
    segments_min_distances_array = np.where(
        segments_min_distances_array < threshold, 1, 0
    )
    segments_time = segments_min_distances_array.argmax(axis=0)

    # Edges activations
    edges_time_interval = {}

    for e in final_graph.edges:
        edge = Edge(Node(e[0], exp), Node(e[1], exp), exp)
        segments_indexes = edges_indexes[f"{edge.begin.label},{edge.end.label}"]
        segments_times = np.array([segments_time[index] for index in segments_indexes])

        begin = np.median(segments_times[:amount_of_border_segment])
        if len(segments_times) > amount_of_border_segment:
            end = np.median(segments_times[-amount_of_border_segment:])
        else:
            end = np.median(segments_times)

        edges_time_interval[f"{edge.begin.label},{edge.end.label}"] = (begin, end)

    model = load_model(model_path)
    positions = exp.positions[last_index]

    # Hypergraph
    G = final_graph
    H = hypergraph_from_graph_ml_2(exp, G, positions, edges_time_interval, model)

    # Save data
    data = {
        "final_graph": final_graph,
        "graph_segemented_final": graph_segemented_final,
        "node_pos": nodes_pos,
        "edges_indexes": edges_indexes,
        "segments_index": segments_index,
        "segments_center_final": segments_center_final,
        "segments_centers": segments_centers,
        "segments_min_distances": segments_min_distances,
        "segments_time": segments_time,
        "edges_time_interval": edges_time_interval,
        "hypergraph": H,
    }
    path_to_save = os.path.join(
        saving_path, f"hypergraph_data_{plates[0]}_from_{first_index}_to_{last_index}.p"
    )
    with open(path_to_save, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # To Change according to your need : path for exp
    directory_targ = r"/Users/amin/Documents/AMOLF/Data/Prince/"
    saving_path = r"/Users/amin/Documents/AMOLF/Data/Examples/test_save_hypergraph"
    plates = ["482_20230908"]
    first_index = 0
    last_index = 20
    # When there is a point at a distance of segment under the threshold, the segment is activated (distance in pixel)
    # A good distance is (2*segments_length)**2
    # Don't forget to square because closest_point give the distance squared
    threshold = 10**2
    # Amount of segment to look for at in an edge to get the date at which the edge encounter the node
    # Depends of how big segments are and what threshold you use
    amount_of_border_segment = 7
    # Input size of the model : length_pixel_list * 2(two edges) * 2(x and y)
    length_pixel_list = 100
    # Path of the model
    model_path = "/Users/amin/Documents/AMOLF/Data/models/intersection_model_2.keras"

    main(
        directory_targ,
        saving_path,
        plates,
        first_index,
        last_index,
        threshold,
        amount_of_border_segment,
        length_pixel_list,
        model_path,
    )
