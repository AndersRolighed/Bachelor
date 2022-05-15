from MethodApproach import *


def hard_coded_experiment_five_plot():
    y1 = [0., 0., 0., 0., 0.02, 0.03, 0.15, 0.21, 0.28, 0.33, 0.43, 0.46, 0.53, 0.54, 0.57, 0.59, 0.61, 0.62, 0.63,
          0.65]
    y2 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01, 0.05, 0.09, 0.16, 0.19, 0.22, 0.24, 0.25, 0.28, 0.39,
          0.48, 0.49, 0.53, 0.57, 0.58, 0.58, 0.59, 0.58, 0.63, 0.62, 0.62, 0.63, 0.63, 0.64, 0.64, 0.64, 0.65, 0.65]
    y3 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.01, 0.12, 0.21,
          0.3, 0.38, 0.42, 0.47, 0.49, 0.54, 0.56, 0.59, 0.61, 0.59, 0.63, 0.63, 0.64, 0.65]
    xdummies = np.arange(1, 41, 1)
    xdummies2 = np.arange(21, 41, 1)

    clusters = np.array([])
    means = np.array([])

    clusters = np.hstack((clusters, np.ones(20)))
    means = np.hstack((means, np.arange(0.05, 1.05, 0.05)))
    clusters2 = np.arange(1, 21, 1)
    clusters2 = [str(x) for x in clusters2]
    clusters2 = [''.join('(' + clusters2[i] + ')') for i in range(len(clusters2))]
    clusters = np.hstack((clusters, clusters2))
    means = np.hstack((means, np.ones(20)))

    clusters = np.hstack((clusters, np.arange(1, 21, 1)))
    means = np.hstack((means, np.full(20, 0.05)))
    clusters = np.hstack((clusters, np.full(20, 20)))
    means = np.round(means, 3)
    means2 = np.arange(0.05, 1.05, 0.05)
    means2 = np.round(means2, 3)
    means2 = [str(x) for x in means2]
    means2 = [''.join('(' + means2[i] + ')') for i in range(len(means2))]
    means = np.hstack((means, means2))

    strClusters = [str(x) for x in clusters]
    strMeans = [str(x) for x in means]
    upperAxis = np.zeros(40)
    upperAxis = [str(x) for x in upperAxis]
    lowerAxis = np.zeros(40)
    lowerAxis = [str(x) for x in lowerAxis]

    j = 0
    for i in range(len(strClusters)):
        if i <= len(strClusters) / 2 - 1:
            lowerAxis[i] = strMeans[i] + ' : ' + strClusters[i]
        else:
            upperAxis[j] = strMeans[i] + ' : ' + strClusters[i]
            j += 1

    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_xticks(xdummies, lowerAxis, rotation=80, size=16)
    ax.tick_params(axis='x', colors='b')

    ax2 = ax.twiny()
    ax2.set_xticks(xdummies, upperAxis, rotation=80, size=16)
    ax2.tick_params(axis='x', colors='red')

    lns1 = ax2.plot(xdummies, y2, label='Set-up #1: Peaking y_mu first')
    lns2 = ax.plot(xdummies, y3, color='r', label='Set-up #2: Peaking cluster_par first')
    lns3 = ax.plot(xdummies2, y1, color='g', label='Set-up #3 Increasing both parameters simultaneously')
    ax.axvline(20.5, linestyle='--')

    ax.set_ylabel('Ratio of 3-d graph above threshold')
    ax2.set_xlabel('Parameters for set-up #3 - y_mu : cluster_par')
    ax.set_xlabel('Parameters for set-up #1 - y_mu : cluster_par ')

    ax.xaxis.label.set_size(30)
    ax2.xaxis.label.set_size(30)
    ax.yaxis.label.set_size(30)
    ax.tick_params(axis='y', labelsize=16)

    lns = lns1 + lns2 + lns3
    labs = [line.get_label() for line in lns]
    ax.legend(lns, labs, loc=0, title='Experiment set-ups:', fontsize=23)

    # ax.legend(title='Experiment set-ups:')
    plt.show()


# cluster_parameters_experiment_method((5, 5, 5), 100, 1, 0.1, 40, 40, 5, True)
def Experiment_6(dimensions, d_size, y_mu, increment, perm, iterations, cluster_par, plot):
    y_mu_list = np.arange(0, y_mu, increment)
    m = cluster_par / (y_mu / increment)
    cluster_size = np.arange(0, cluster_par, m)
    signi = np.zeros((len(y_mu_list), len(cluster_size)))
    k = 0
    for mu in y_mu_list:
        j = 0
        for size in cluster_size:
            sig = 0
            for i in range(iterations):
                g = create_grid_graph((dimensions[0], dimensions[1], dimensions[2]), d_size, 0, 1, mu, 1, size)
                sig = sig + method(g, 0.05, 'data', 100, perm)
            signi[k, j] = sig / iterations
            j = j + 1
        k = k + 1
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


def experiment_7():
    edge_ratios = [0.01, 0.04, 0.07, 0.1, 0.5, 0.65, 0.85, 1.0]
    edge_neighbors = [1, 3, 6, 9, 12, 15, 18, 21]

    iterations = 200
    perc = np.zeros(3)
    perc_random = np.zeros(len(edge_ratios))
    perc_PA = np.zeros(len(edge_neighbors))
    j = 0
    for i in range(iterations):
        g = create_circle_graph(125, 100, 0, 1, 0.2, 1, 5)
        perc[j] = perc[j] + method(g, 0.05, 'data', 100, 100)
    j = j + 1
    print('Done with circular graphs')
    for i in range(iterations):
        g = create_grid_graph((25, 5, 1), 100, 0, 1, 0.2, 1, 5)
        perc[j] = perc[j] + method(g, 0.05, 'data', 100, 100)
    j = j + 1
    print('Done with 2-d grids')
    for i in range(iterations):
        g = create_grid_graph((5, 5, 5), 100, 0, 1, 0.2, 1, 5)
        perc[j] = perc[j] + method(g, 0.05, 'data', 100, 100)
    j = j + 1
    print('Done with 3-d grids')
    j = 0
    for edges in edge_ratios:
        for i in range(iterations):
            g = simulate_network(100, 'random', edges, 0, 0, 1, 0.2, 1, 100, 5)
            perc_random[j] = perc_random[j] + method(g, 0.05, 'data', 100, 100)
        j = j + 1
        print('Done with ', j, ' / ', len(edge_ratios))
    print('Done with Random graphs')
    j = 0
    for edges in edge_neighbors:
        for i in range(iterations):
            g = simulate_network(100, 'PA', 0, edges, 0, 1, 0.2, 1, 100, 5)
            perc_PA[j] = perc_PA[j] + method(g, 0.05, 'data', 100, 100)
        j = j + 1
        print('Done with ', j, ' / ', len(edge_neighbors))
    print('Done with PA graphs')
    print('Percentages for preliminary graphs: ', perc / iterations)
    print('Percentages for random graphs: ', perc_random / iterations)
    print('Percentages for PA graphs: ', perc_PA / iterations)


if __name__ == '__main__':
    print()
