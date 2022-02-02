
# imports
import numpy as np
import matplotlib.pyplot as plt


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
    return np.mean(first)-np.mean(last)


def plot_hist(estimates, base_diff, test_value):
    diff = estimates-base_diff
    n, bins, patches = plt.hist(diff, 100, density=True, facecolor='g', alpha=0.75)

    a = min(diff)
    b = max(n)
    rounded = round(test_value, 3)

    plt.xlabel('Difference in mean')
    plt.ylabel('Density')
    plt.title('Histogram of difference of means of permuted groupings')
    plt.text(a, b, 'P-value ='+str(rounded))
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    x_data, y_data = get_data('norm', 'norm', 0, 0, 1, 1, 100) #distributionx, distributiony, meanx, meany, variancex, variancey, size
    base_diff = np.mean(x_data)-np.mean(y_data)
    print('Base difference in means = ', base_diff)

    pooled = np.hstack([x_data, y_data])
    num_samples = 1000
    estimates = list(map(lambda x: run_permutation(pooled, x_data.size), range(num_samples)))
    count = len(np.where(estimates <= base_diff)[0])
    test_value = 1.0 - (float(count)/float(num_samples))
    print('P-value = ', test_value)

    plot_hist(estimates, base_diff, test_value)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
