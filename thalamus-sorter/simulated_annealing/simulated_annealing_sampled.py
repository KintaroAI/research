def cost_function_sampled(weight_matrix, neurons_count, cols, sample_size):
    """Choose random points on the matrix to compute cost."""
    cost = 0
    for _ in range(sample_size):
        i, j = random.sample(range(neurons_count), 2)
        x1, y1 = i % cols, i // cols
        x2, y2 = j % cols, j // cols
        dist = abs(x1 - x2) + abs(y1 - y2)
        weight = weight_matrix[i, j]
        cost += weight * dist
    return cost
