# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


def experiment_one(x_dist, y_dist, x_mu, y_mu, x_sigma, y_sigma,
                   size, iterations, significance, p_toggle):
    if p_toggle:
        print("## Running experiment one ## ")
        print(" ")

    plot = True
    significant_iterations = 0
    for i in range(iterations):
        x_data, y_data = get_data(x_dist, y_dist, x_mu, y_mu,
                                  x_sigma, y_sigma, size)
        base_diff = np.mean(x_data) - np.mean(y_data)
        pooled = np.hstack([x_data, y_data])
        num_samples = 1000
        estimates = list(map(lambda x: run_permutation(pooled, x_data.size),
                             range(num_samples)))
        count = len(np.where(np.abs(estimates) <= np.abs(base_diff))[0])
        test_value = 1.0 - ((1.0 + float(count)) / (float(num_samples) + 1.0))

        if test_value <= float(significance):
            significant_iterations += 1
            if p_toggle and plot:
                plot_hist(estimates, base_diff, test_value, False)
                plot = False

    if p_toggle:
        print("Ran experiment one and got the following false alarm rate: ",
              ((significant_iterations + 1) / (iterations + 1)))
        plt.axvline(x=0, color='r', linestyle='-')
        plt.show()
    return (significant_iterations + 1) / (iterations + 1)


def experiment_two(x_dist, y_dist, x_mu, y_mu, x_sigma, y_sigma,
                   iterations, significance, p_toggle):
    print("## Running experiment two ##")
    print(" ")

    distributions = np.linspace(y_mu, x_mu, 150)
    count = 0
    sizes = [50, 100, 200, 500]
    colors = ['g', 'b', 'm', 'y']
    i = 0
    for number in sizes:
        p_values = []
        for value in distributions:
            p_value = experiment_one(x_dist, y_dist, value, y_mu, x_sigma, y_sigma,
                                     number, iterations, significance,
                                     p_toggle)
            p_values.append(p_value)
            count += 1
            print("Ran iteration ", count, "/", len(distributions))
        count = 0
        plt.plot(p_values, distributions, color=colors[i])
        i += 1

    plt.xlabel('Mean of X-distribution')
    plt.axhline(y=0.05, color='r', linestyle='-')
    plt.ylabel('Level of significance')
    plt.title('Different sample sizes and the significant '
              'difference between groups')
    plt.grid(True)
    plt.show()


def experiment_two_5(x_dist, y_dist, x_mu, y_mu, x_sigma, y_sigma,
                     iterations, significance, p_toggle):
    print("## Running experiment two ##")
    print(" ")

    distributions = np.linspace(y_mu, x_mu, 300)
    count = 0
    p_values = []
    for value in distributions:
        p_value = experiment_one(x_dist, y_dist, value, y_mu, x_sigma, y_sigma,
                                 100, iterations, significance,
                                 p_toggle)
        p_values.append(p_value)
        count += 1
        print("Ran iteration ", count, "/", len(distributions))
    plt.plot(p_values, distributions)

    plt.xlabel('Mean of X-distribution')
    plt.axhline(y=0.05, color='r', linestyle='-')
    plt.ylabel('Level of significance')
    plt.title('Significance between x, and y with mean = 0')
    plt.text(distributions[0] + 0.04, p_values[len(distributions) - 1] / 2,
             'p(X0) =' + str(round(p_values[0], 5)))
    plt.grid(True)
    plt.show()


def experiment_three_tester(x_dist, y_dist, x_mu, y_mu, x_sigma, y_sigma, size,
                            iterations, significance, attributes, p_toggle):
    if p_toggle:
        print("## Running experiment three with ", attributes, " attributes ## ")
        print(" ")

    significance_cor = significance / attributes

    graph_signi = []
    graph_signi_cor = []
    significant_iterations_cor = 0
    significant_iterations = 0
    for j in range(attributes):
        for i in range(iterations):
            x_data, y_data = get_data(x_dist, y_dist, x_mu, y_mu, x_sigma,
                                      y_sigma, size)
            base_diff = np.mean(x_data) - np.mean(y_data)
            pooled = np.hstack([x_data, y_data])
            num_samples = 1000
            estimates = list(map(lambda x: run_permutation(pooled, x_data.size),
                                 range(num_samples)))
            count = len(np.where(estimates <= base_diff)[0])
            test_value = 1.0 - ((1.0 + float(count)) / (float(num_samples) + 1.0))

            if test_value <= float(significance):
                significant_iterations += 1

            if test_value <= float(significance_cor):
                significant_iterations_cor += 1

        graph_signi.append(significant_iterations)
        graph_signi_cor.append(significant_iterations_cor)
        print("Finished comparing ", j + 1, "/", attributes, " attributes")

    if p_toggle:
        print("Ran experiment three and got the following false alarm rate: ",
              ((significant_iterations + 1) / (iterations + 1)))
        print("When correcting with bonferroni we get the following false alarm rate: ",
              ((significant_iterations_cor + 1) / (iterations + 1)))

    return (significant_iterations + 1) / (iterations + 1), \
           (significant_iterations_cor + 1) / (iterations + 1), \
           graph_signi, graph_signi_cor


def experiment_three(x_dist, y_dist, x_mu, y_mu, x_sigma, y_sigma, size,
                     iterations_outer, significance, null_hypotheses):
    print("## Running experiment five ##")
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
            temp_p = []
            for j in range(node):
                x_data, y_data = get_data(x_dist, y_dist, x_mu, y_mu,
                                          x_sigma, y_sigma, size)
                base_diff = np.mean(x_data) - np.mean(y_data)
                pooled = np.hstack([x_data, y_data])
                num_samples = 1000
                estimates = list(
                    map(lambda x: run_permutation(pooled, x_data.size),
                        range(num_samples)))
                count = len(np.where(estimates <= base_diff)[0])
                test_value = 1.0 - ((1.0 + float(count)) /
                                    (float(num_samples) + 1.0))

                temp_p.append(test_value)

            if any(x < significance for x in temp_p):
                significant_iterations[node - 1] += 1
            if any(x < significance / node for x in temp_p):
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


def conf_interval(iterations, conf_level):
    p_values = []
    for i in range(iterations):
        p_values.append(
            experiment_one('norm', 'norm', 0, 0, 1, 1, 100, 1000, 0.05, False))
        print('done with', i + 1, '/', iterations)
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
    return conf_int


def get_data(x_dist, y_dist, x_mu, y_mu, x_sigma, y_sigma, size):
    # np.random.seed(19680801)
    x_data = None
    y_data = None
    if x_dist == 'norm':
        x_data = np.random.normal(x_mu, x_sigma, size)
    if y_dist == 'norm':
        y_data = np.random.normal(y_mu, y_sigma, size)
    if x_dist == 'random':
        x_data = np.random.random(size)
    if y_dist == 'random':
        y_data = np.random.random(size)
    return x_data, y_data


def run_permutation(pooled, size):
    np.random.shuffle(pooled)
    first = pooled[:size]
    last = pooled[-size:]
    return np.mean(first) - np.mean(last)


def plot_hist(estimates, base_diff, test_value, show):
    diff = estimates - base_diff
    n, bins, patches = \
        plt.hist(diff, 100, density=True, facecolor='g', alpha=0.75)

    a = min(diff)
    b = max(n)
    rounded = round(test_value, 4)

    plt.xlabel('Difference in mean')
    plt.ylabel('Density')
    plt.title('Histogram of difference of means of permuted groupings')
    plt.text(a, b, 'P-value =' + str(rounded))
    plt.grid(True)
    if show:
        plt.show()


if __name__ == '__main__':
    print()
