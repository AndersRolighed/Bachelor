import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Grid import create_grid_graph
from Circle import create_circle_graph
from matplotlib.ticker import LinearLocator
from matplotlib import cm
from numba import jit, cuda


# @jit(target ="cuda")
def method(graph, significance, data, data_size, permutations):
    # run initial time
    t_stat_original = find_t_stat(graph, significance, data, data_size, False)

    # run iterations number of permutations
    t_stats = []
    for i in range(permutations):
        t_stats.append(find_t_stat(graph, significance, data, data_size, True))

    percentiles = [2.50, 97.50]
    first, last = np.percentile(t_stats, percentiles)
    if t_stat_original <= first or t_stat_original >= last:
        print("Graph has undergone change")
        return 1
    else:
        print("No change detected in the graph")
        return 0


def plot_graph(graph):
    # 3d spring layout
    significant_nodes = []
    for node in sorted(graph):
        p = get_node_attribute(graph, node, 'p', 'data', False, 100)
        if p <= 0.05:
            significant_nodes.append(node)
    pos = nx.spring_layout(graph, dim=3, seed=779)
    # Extract node and edge positions from the layout
    colored_nodes = np.array([pos[v] for v in significant_nodes])
    node_xyz = np.array([pos[v] for v in sorted(graph) if v not in colored_nodes])
    edge_xyz = np.array([(pos[u], pos[v]) for u, v in graph.edges()])

    # Create the 3D figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the nodes - alpha is scaled by "depth" automatically
    ax.scatter(*node_xyz.T, s=100, ec="w", alpha=0.3)
    ax.scatter(*colored_nodes.T, s=100, ec="w", color='r', alpha=1)

    # Plot the edges
    for vizedge in edge_xyz:
        ax.plot(*vizedge.T, color="tab:gray")

    def _format_axes(ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])

    _format_axes(ax)
    fig.tight_layout()
    plt.show()


def find_t_stat(graph, significance, data, data_size, perm_trigger):
    clusters_list = []
    checked_nodes = []
    for node in sorted(graph):
        if node in checked_nodes:
            pass
        else:
            p_value, t_stat = get_node_attribute(graph, node, attribute='both', data_name=data, permute=perm_trigger,
                                                 d_size=data_size)
            if p_value > significance:
                checked_nodes.append(node)
            else:
                cluster = t_stat
                # list of nodes missing being searched
                missing_nodes = [node]
                while len(missing_nodes) > 0:
                    neighbors = list(graph.neighbors(missing_nodes.pop()))
                    # loop through neighbors
                    for neighbor in neighbors:
                        if neighbor in checked_nodes:
                            pass
                        else:
                            if get_node_attribute(graph, neighbor, attribute='p', data_name=data, permute=perm_trigger,
                                                  d_size=data_size) < significance:
                                cluster = cluster + get_node_attribute(graph, neighbor, attribute='t', data_name=data,
                                                                       permute=perm_trigger, d_size=data_size)
                                missing_nodes.append(neighbor)
                        checked_nodes.append(neighbor)
                clusters_list.append(np.abs(cluster))
    # print(clusters_list)
    if clusters_list:
        return max(clusters_list)
    else:
        return 0


def get_node_attribute(graph, node, attribute, data_name, permute, d_size):
    if permute:
        data = graph.nodes[node][data_name]
        np.random.shuffle(data)
        t_stat, p_value = stats.ttest_ind(data[:d_size], data[d_size:])
        if attribute == 'both':
            return p_value, t_stat
        if attribute == 'p':
            return p_value
        if attribute == 't':
            return t_stat
    else:
        data = graph.nodes[node][data_name]
        t_stat, p_value = stats.ttest_ind(data[:d_size], data[d_size:])

        if attribute == 'both':
            return p_value, t_stat
        if attribute == 'p':
            return p_value
        if attribute == 't':
            return t_stat


def simulate_network(nodes, network_type, prob, m, x_mu, x_sigma, y_mu, y_sigma, d_size, cluster_par):
    if network_type != 'random' and network_type != 'PA':
        print('Not one of the built-in types. Do "random" for a random graph  or "PA" for preferential attachment')
        pass
    else:
        if network_type == 'random':
            graph = nx.erdos_renyi_graph(nodes, prob, seed=111)
        if network_type == 'PA':
            graph = nx.barabasi_albert_graph(nodes, m, seed=111)
        for node in sorted(graph):
            x_data = np.random.normal(x_mu, x_sigma, d_size)
            y_data = np.random.normal(x_mu, x_sigma, d_size)
            data = np.hstack([x_data, y_data])
            graph.nodes[node]['data'] = data

        count = 0
        if count < cluster_par:
            graph_list = sorted(graph)
            trigger = True
            while trigger:
                random_node = graph_list[np.random.randint(len(graph_list))]
                neighbors_list = list(graph.neighbors(random_node))
                if neighbors_list:
                    trigger = False
            x_data = np.random.normal(x_mu, x_sigma, d_size)
            y_data = np.random.normal(y_mu, y_sigma, d_size)
            data = np.hstack([x_data, y_data])
            graph.nodes[random_node]['data'] = data
            count = count + 1
            neighbors_list = list(graph.neighbors(random_node))
            altered_nodes = [random_node]

            while count < cluster_par:
                next_node = neighbors_list[np.random.randint(len(neighbors_list))]
                node_neighbors = graph.neighbors(next_node)

                for node in node_neighbors:
                    if node not in neighbors_list and node not in altered_nodes:
                        neighbors_list.append(node)

                x_data = np.random.normal(x_mu, x_sigma, d_size)
                y_data = np.random.normal(y_mu, y_sigma, d_size)
                data = np.hstack([x_data, y_data])
                graph.nodes[next_node]['data'] = data
                count = count + 1

    return graph


if __name__ == '__main__':
    print()
