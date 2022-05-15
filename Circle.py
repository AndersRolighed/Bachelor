import networkx as nx
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import t
import matplotlib.pyplot as plt
from scipy import stats


def cluster_permutation_test(g_size, d_size, significance, perm, iterations, cluster_par, y_mu):
    null_h_rejects = 0
    for j in range(iterations):
        graph = create_circle_graph(g_size, d_size, 0, 1, y_mu, 1, cluster_par)
        l_cluster_list = []
        for k in range(g_size):
            t_stat, p_value = stats.ttest_ind(graph.nodes[k]['data'][:d_size], graph.nodes[k]['data'][d_size:])
            graph.nodes[k]['p_value'] = p_value
            graph.nodes[k]['t_stat'] = t_stat
        # plot_circle(graph, 0.05)
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


def cluster_parameters_experiment(g_size, d_size, y_mu, increment, perm, iterations, cluster_par, plot):
    y_mu_list = np.arange(0, y_mu, increment)
    m = cluster_par / (y_mu / increment)
    cluster_size = np.arange(0, cluster_par, m)
    signi = np.zeros((len(y_mu_list), len(cluster_size)))
    k = 0
    for mu in y_mu_list:
        j = 0
        for size in cluster_size:
            signi[k, j] = cluster_permutation_test(g_size, d_size, 0.05, perm, iterations, size, mu)
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


def experiment_five(g_size, d_size, perm, iterations):
    clusters = np.array([])
    means = np.array([])
    # Eks 1
    clusters = np.hstack((clusters, np.arange(1, 21, 1)))
    means = np.hstack((means, np.arange(0.05, 1.05, 0.05)))
    # Eks 2
    clusters = np.hstack((clusters, np.ones(20)))
    means = np.hstack((means, np.arange(0.05, 1.05, 0.05)))
    clusters = np.hstack((clusters, np.arange(1, 21, 1)))
    means = np.hstack((means, np.ones(20)))
    # Eks 3
    clusters = np.hstack((clusters, np.arange(1, 21, 1)))
    means = np.hstack((means, np.full(20, 0.05)))
    clusters = np.hstack((clusters, np.full(20, 21)))
    means = np.hstack((means, np.arange(0.05, 1.05, 0.05)))

    percentages1 = np.zeros(len(clusters))

    # perm = 30, iterations = 30, g_size = 100, d_size = 100
    for i in range(len(clusters) - 1):
        sig, per = cluster_parameters_experiment(g_size, d_size, means[i], means[i] / 10, perm, iterations, clusters[i])
        percentages1[i] = per

    print("Percentages:", percentages1)
    return percentages1


def cluster_parameters_experiment_1_par(g_size, d_size, y_mu, increment, perm, iterations, cluster_par):
    if y_mu > 0:
        y_mu_list = np.arange(0.05, y_mu, increment)
        signi = np.zeros(len(y_mu_list))
        k = 0
        for mu in y_mu_list:
            signi[k] = cluster_permutation_test(g_size, d_size, 0.05, perm, iterations, 1, mu)
            k += 1

        # plt.plot(y_mu_list, signi)
        # plt.xlabel('Mu of y distribution')
        # plt.ylabel('Significance')
        # plt.show()

    if cluster_par > 0:
        cluster_size = np.arange(1, cluster_par, 1)
        signi = np.zeros(len(cluster_size))
        k = 0
        for size in cluster_size:
            signi[k] = cluster_permutation_test(g_size, d_size, 0.05, perm, iterations, size, 1.0)
            k += 1

        # plt.plot(cluster_size, signi)
        # plt.xlabel('Cluster size')
        # plt.ylabel('Significance')
        # plt.show()

    per = ((signi > 0.40).sum()) / len(signi)
    return signi, per


def update_cluster_list(graph, cluster_list):
    t_values = np.zeros(len(cluster_list))
    j = 0
    for cluster in cluster_list:
        t_sum = 0
        if type(cluster) == list:
            for i in cluster:
                t_sum += abs(get_node_attribute(graph, i, 't_stat'))
        else:
            t_sum += abs(get_node_attribute(graph, cluster, 't_stat'))
        t_values[j] += t_sum
        j += 1
    return t_values


def find_cluster(graph, significance):
    clusters_list = []
    for i in range(len(graph.nodes)):
        if any(i in groups for groups in clusters_list):
            pass
        else:
            cluster = []
            if get_node_attribute(graph, i, 'p_value') < significance:
                cluster.append(i)

                # list of nodes missing being searched
                missing_nodes = [i]

                while len(missing_nodes) > 0:
                    neighbors = list(graph.neighbors(missing_nodes.pop()))
                    # loop through neighbors

                    for neighbor in neighbors:
                        if get_node_attribute(graph, neighbor, 'p_value') < significance:
                            cluster.append(neighbor)
                            if any(neighbor == searched_neighbors for searched_neighbors in cluster):
                                pass
                            else:
                                missing_nodes.append(neighbor)
                clusters_list.append(cluster)
    return clusters_list


def test_significance(significance, iterations, g_size, d_size, conf_level):
    p_values = []
    for j in range(iterations):
        circle_graph = create_circle_graph(g_size, d_size, 0, 1, 0, 1, 0)
        update_p_values(circle_graph, d_size)
        significant = 0
        for i in range(len(circle_graph.nodes)):
            temp_p = get_node_attribute(circle_graph, i, 'p_value')
            if temp_p <= significance:
                significant += 1
        p_values.append(significant / len(circle_graph.nodes))
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


def circle_hypotheses(d_size, x_mu, y_mu, x_sigma, y_sigma,
                      iterations_outer, significance, null_hypotheses):
    print("## Running circle_hypotheses experiment ##")
    print(" ")
    nodes = np.arange(1, null_hypotheses, 1)
    significant_iterations = np.zeros(len(nodes))

    significant_iterations_bonf = np.zeros(len(nodes))

    # Perform the loop 1000 times so to be able to say something with
    # statistical confidence
    for i in range(iterations_outer):
        # Perform experiment one a different amount of times to see how
        # often it guarantees a significant result
        for node in nodes:
            circle_graph = create_circle_graph(node, d_size, x_mu, x_sigma, y_mu, y_sigma, 0)
            update_p_values(circle_graph, d_size)
            temp_p = []
            for j in range(len(circle_graph.nodes)):
                temp_p.append(get_node_attribute(circle_graph, j, 'p_value'))

            if any(p < significance for p in temp_p):
                significant_iterations[node - 1] += 1
            if any(p < significance / node for p in temp_p):
                significant_iterations_bonf[node - 1] += 1

        print("done with ", i + 1, "/", iterations_outer)

    plt.axhline(y=0.05, color='m', linestyle='-')
    plt.plot(nodes, 1 - (1 - significance) ** nodes)
    plt.plot(nodes, [(iteration + 1) / (iterations_outer + 1)
                     for iteration in significant_iterations_bonf], color='r')
    plt.plot(nodes, [(iteration + 1) / (iterations_outer + 1)
                     for iteration in significant_iterations], color='g')
    plt.xlabel('Number of independent null hypotheses')
    plt.ylabel('Chance of type I error')
    plt.title('Chance of type I error when introducing more null hypotheses')
    plt.grid(True)
    plt.show()


def plot_circle(graph, significance):
    significant_nodes = []
    nodes = []
    pos = nx.circular_layout(graph)
    t_stats = []
    for i in range(len(graph.nodes)):
        if get_node_attribute(graph, i, 'p_value') <= significance:
            significant_nodes.append(i)
            t_stats.append(get_node_attribute(graph, i, 't_stat'))
        else:
            nodes.append(i)
    options = {"edgecolors": "tab:gray", "node_size": 5000 / len(graph.nodes), "alpha": 0.9}
    nx.draw_networkx_nodes(graph, pos, nodelist=significant_nodes, node_color="tab:red", **options)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color="tab:blue", **options)
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def t_permutations(pooled, x_data_size):
    np.random.shuffle(pooled)
    t_stat, p_value = stats.ttest_ind(pooled[:x_data_size], pooled[x_data_size:])
    return t_stat, p_value


def update_p_values(graph, x_data_size):
    for i in range(len(list(graph.nodes))):
        pooled = get_node_attribute(graph, i, 'data')
        t_stat, p_value = t_permutations(pooled, x_data_size)

        graph.nodes[i]['p_value'] = p_value
        graph.nodes[i]['t_stat'] = t_stat
    return graph


def create_circle_graph(g_size, d_size, x_mu, x_sigma, y_mu, y_sigma, cluster_par):
    graph = nx.cycle_graph(g_size)
    for i in range(g_size):
        if i < cluster_par:
            x_data = np.random.normal(x_mu, x_sigma, d_size)
            y_data = np.random.normal(y_mu, y_sigma, d_size)
            data = np.hstack([x_data, y_data])
            graph.nodes[i]['data'] = data
        else:
            x_data = np.random.normal(x_mu, x_sigma, d_size)
            y_data = np.random.normal(x_mu, x_sigma, d_size)
            data = np.hstack([x_data, y_data])
            graph.nodes[i]['data'] = data
    return graph


def get_node_attribute(graph, node, attribute):
    data = graph.nodes[node][attribute]
    return data


if __name__ == '__main__':
    print()
