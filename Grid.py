import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy import stats
from mpl_toolkits.mplot3d import axes3d


def cluster_permutation_test_grid(dimensions, d_size, significance, perm, iterations, cluster_par, y_mu, plot):
    null_h_rejects = 0
    for j in range(iterations):
        graph = create_grid_graph(dimensions, d_size, 0, 1, y_mu, 1, cluster_par)
        l_cluster_list = []
        for node in sorted(graph):
            t_stat, p_value = stats.ttest_ind(graph.nodes[node]['data'][:d_size], graph.nodes[node]['data'][d_size:])
            graph.nodes[node]['p_value'] = p_value
            graph.nodes[node]['t_stat'] = t_stat
        plot_graph(graph)
        cluster_list_og = find_cluster(graph, significance)
        t_stat_clusters_og = update_cluster_list(graph, cluster_list_og)
        if len(cluster_list_og) > 0:
            l_cluster_og = (max(abs(t_stat_clusters_og)))

            for i in range(perm):
                graph = update_p_values(graph, d_size)
                cluster_list = find_cluster(graph, significance)
                t_stat_clusters = update_cluster_list(graph, cluster_list)
                if len(cluster_list) > 0:
                    l_cluster_list.append(max(abs(t_stat_clusters)))

            percentiles = [2.50, 97.50]
            first, last = np.percentile(l_cluster_list, percentiles)
            if l_cluster_og <= first or l_cluster_og >= last:
                null_h_rejects += 1
            print('Done with ', j + 1, '/', iterations)
    if plot:
        plt.hist(l_cluster_list, 50, density=True, facecolor='g', alpha=0.75)
        for q in np.percentile(l_cluster_list, percentiles):
            plt.axvline(q, color='orange')
        plt.axvline(l_cluster_og, color='r')
        plt.xlabel('Sum of t-value within largest cluster')
        plt.ylabel('Density')
        plt.title('Histogram of largest t-values at each permutation')
        plt.show()
        print("ran a cluster permutation test")
    return (null_h_rejects + 1) / (iterations + 1)


def find_cluster(graph, significance):
    clusters_list = []
    for node in sorted(graph):
        if any(node in groups for groups in clusters_list):
            pass
        else:
            cluster = []
            if get_node_attribute_grid(graph, node, 'p_value') < significance:
                cluster.append(node)

                # list of nodes missing being searched
                missing_nodes = [node]

                while len(missing_nodes) > 0:
                    neighbors = list(graph.neighbors(missing_nodes.pop()))
                    # loop through neighbors

                    for neighbor in neighbors:
                        if get_node_attribute_grid(graph, neighbor, 'p_value') < significance:
                            cluster.append(neighbor)
                            if any(neighbor == searched_neighbors for searched_neighbors in cluster):
                                pass
                            else:
                                missing_nodes.append(neighbor)
                clusters_list.append(cluster)
    return clusters_list


def update_cluster_list(graph, cluster_list):
    t_values = np.zeros(len(cluster_list))
    j = 0
    for cluster in cluster_list:
        t_sum = 0
        if type(cluster) == list:
            for node in cluster:
                t_sum += abs(get_node_attribute_grid(graph, node, 't_stat'))
        else:
            t_sum += abs(get_node_attribute_grid(graph, cluster, 't_stat'))
        t_values[j] += t_sum
        j += 1
    return t_values


def cluster_parameters_experiment_grid(dimensions, d_size, y_mu, increment, perm, iterations, cluster_par, plot):
    y_mu_list = np.arange(0, y_mu, increment)
    m = cluster_par / (y_mu / increment)
    cluster_size = np.arange(0, cluster_par, m)
    signi = np.zeros((len(y_mu_list), len(cluster_size)))
    k = 0
    for mu in y_mu_list:
        j = 0
        for size in cluster_size:
            signi[k, j] = cluster_permutation_test_grid(dimensions, d_size, 0.05, perm, iterations, size, mu, False)
            j += 1
        k += 1
        print('Ran cluster parameters experiment ', k, '/', len(y_mu_list), 'time(s)')

    if plot:
        y_mesh, cluster_mesh = np.meshgrid(cluster_size, y_mu_list)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(cluster_mesh, y_mesh, signi, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')

        ax.set_ylabel('Cluster size')
        ax.set_xlabel('y_mu')
        ax.set_zlabel('Significance')
        ax.set_zlim(0, 1.00)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
    print((signi > 0.40).sum())
    per = ((signi > 0.40).sum()) / (signi.shape[0] * signi.shape[1])
    return signi, per


def grid_hypotheses(d_size, x_mu, y_mu, x_sigma, y_sigma,
                    iterations_outer, significance):
    print("## Running circle_hypotheses experiment ##")
    print(" ")
    grid_dimensions = [(1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 2, 1), (5, 1, 1), (3, 2, 1), (7, 1, 1), (2, 2, 2),
                       (3, 3, 1), (5, 2, 1), (11, 1, 1), (3, 2, 2), (13, 1, 1), (7, 2, 1), (5, 3, 1), (4, 2, 2),
                       (17, 1, 1), (3, 3, 2), (19, 1, 1), (5, 2, 2), (7, 3, 1)]

    grid_number = np.arange(1, len(grid_dimensions) + 1, 1)
    print(grid_number)
    significant_iterations = np.zeros(len(grid_number))

    significant_iterations_bonf = np.zeros(len(grid_number))

    # Perform the loop 1000 times so to be able to say something with
    # statistical confidence
    for i in range(iterations_outer):
        # Perform experiment one a different amount of times to see how
        # often it guarantees a significant result
        grid_count = 1
        for grid in grid_dimensions:
            grid_graph = create_grid_graph(grid, d_size, x_mu, x_sigma, y_mu, y_sigma, 0)
            update_p_values(grid_graph, d_size)
            temp_p = []
            for node in sorted(grid_graph):
                temp_p.append(get_node_attribute_grid(grid_graph, node, 'p_value'))

            if any(p < significance for p in temp_p):
                significant_iterations[grid_count - 1] += 1
            if any(p < significance / grid_count for p in temp_p):
                significant_iterations_bonf[grid_count - 1] += 1
            grid_count += 1

        print("done with ", i + 1, "/", iterations_outer)

    plt.axhline(y=0.05, color='m', linestyle='-')
    plt.plot(grid_number, 1 - (1 - significance) ** grid_number)
    plt.plot(grid_number, [(iteration + 1) / (iterations_outer + 1)
                           for iteration in significant_iterations_bonf], color='r')
    plt.plot(grid_number, [(iteration + 1) / (iterations_outer + 1)
                           for iteration in significant_iterations], color='g')
    plt.xlabel('Number of independent null hypotheses')
    plt.ylabel('Chance of type I error')
    plt.title('Chance of type I error when introducing more null hypotheses')
    plt.grid(True)
    plt.show()


def test_significance(significance, iterations, dimensions, d_size, conf_level):
    p_values = []
    for j in range(iterations):
        grid_graph = create_grid_graph(dimensions, d_size, 0, 1, 0, 1, 0)
        update_p_values(grid_graph, d_size)
        significant = 0
        for node in sorted(grid_graph):
            temp_p = get_node_attribute_grid(grid_graph, node, 'p_value')
            if temp_p <= significance:
                significant += 1
        p_values.append(significant / len(grid_graph.nodes))
    SEM = np.std(p_values, ddof=1) / np.sqrt(np.size(p_values))
    sd = np.std(p_values)
    mean = np.mean(p_values)
    dof = len(p_values) - 1
    t_criterion = np.abs(t.ppf((1 - conf_level) / 2, dof))
    conf_int = [(mean - sd * t_criterion / np.sqrt(len(p_values)),
                 mean + sd * t_criterion / np.sqrt(len(p_values)))]
    print("Standard error of the mean = ", SEM)
    print("mean = ", mean)
    print("Confidence interval = ", conf_int)


def update_p_values(graph, x_data_size):
    for node in sorted(graph):
        pooled = get_node_attribute_grid(graph, node, 'data')
        t_stat, p_value = t_permutations(pooled, x_data_size)

        graph.nodes[node]['p_value'] = p_value
        graph.nodes[node]['t_stat'] = t_stat
    return graph


def t_permutations(pooled, x_data_size):
    np.random.shuffle(pooled)
    t_stat, p_value = stats.ttest_ind(pooled[:x_data_size], pooled[x_data_size:])
    return t_stat, p_value


def plot_graph(graph):
    # 3d spring layout
    significant_nodes = []
    for node in sorted(graph):
        p = get_node_attribute_grid(graph, node, 'p_value')
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


def create_grid_graph(dimensions, d_size, x_mu, x_sigma, y_mu, y_sigma, cluster_par):
    graph = nx.grid_graph(dim=(dimensions[0], dimensions[1], dimensions[2]))
    for node in sorted(graph):
        x_data = np.random.normal(x_mu, x_sigma, d_size)
        y_data = np.random.normal(x_mu, x_sigma, d_size)
        data = np.hstack([x_data, y_data])
        graph.nodes[node]['data'] = data

    count = 0
    if count < cluster_par:
        graph_list = sorted(graph)
        random_node = graph_list[np.random.randint(len(graph_list))]
        x_data = np.random.normal(x_mu, x_sigma, d_size)
        y_data = np.random.normal(y_mu, y_sigma, d_size)
        data = np.hstack([x_data, y_data])
        graph.nodes[random_node]['data'] = data
        count += 1
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
            count += 1

    return graph


def get_node_attribute_grid(graph, node, attribute):
    data = graph.nodes[(node[0], node[1], node[2])][attribute]
    return data


if __name__ == '__main__':
    print()
