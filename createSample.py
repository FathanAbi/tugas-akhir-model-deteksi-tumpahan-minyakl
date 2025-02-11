import numpy as np
import random
import zeroPadding

def createSample(hsi, patch_size, num_per_class):
    training_hsi = hsi
    training_hsi_gt = training_hsi.gt
    print("hsi shape")
    print(training_hsi.img.shape)

    # Get indices of elements equal to 0
    indices_0 = np.where(training_hsi_gt == 0)
    indices_1 = np.where(training_hsi_gt == 1)

    # Convert to a list of (row, col) tuples
    zero_indices = list(zip(indices_0[0], indices_0[1]))

    one_indices = list(zip(indices_1[0], indices_1[1]))

    x = len(zero_indices)
    y = len(one_indices)

    random_indices_0 = random.sample(zero_indices, k=num_per_class)
    print("5 Randomly chosen 0 indices:", random_indices_0)

    random_indices_1 = random.sample(one_indices, k=num_per_class)
    print("5 Randomly chosen 1 indices:", random_indices_1)

    data = training_hsi.img
    patch = patch_size
    n_bands = 224
    half_patch = patch // 2

    # for class 0
    n = len(random_indices_0)
    n2 = len(random_indices_1)

    selected_patch_0=np.zeros((n, patch, patch, n_bands))
    selected_patch_1=np.zeros((n2, patch, patch, n_bands))

   

    matrix=zeroPadding.zeroPadding_3D(data,half_patch) #add 0 in every side of the data
    print(data.shape)
    print(matrix.shape)

    for i in range (n): #if padded the index are changing
        # x_pos = random_indices_0[i][0] - half_patch
        # y_pos = random_indices_0[i][1] - half_patch
        x_pos = random_indices_0[i][0]
        y_pos = random_indices_0[i][1] 
        selected_rows = matrix[range(x_pos,x_pos+2*half_patch+1), :]
        selected_patch_0[i] = selected_rows[:, range(y_pos, y_pos+2*half_patch+1)]

        print(x_pos, y_pos, random_indices_0[i][0], random_indices_0[i][1])

    for i in range (n): #if padded the index are changing
        # x_pos = random_indices_0[i][0] - half_patch
        # y_pos = random_indices_0[i][1] - half_patch
        x_pos = random_indices_1[i][0]
        y_pos = random_indices_1[i][1] 
        selected_rows = matrix[range(x_pos,x_pos+2*half_patch+1), :]
        selected_patch_1[i] = selected_rows[:, range(y_pos, y_pos+2*half_patch+1)]

        print(x_pos, y_pos, random_indices_1[i][0], random_indices_1[i][1])

    i = 0
    

    print("seed pixel in data class 0")
    print(data[random_indices_0[i][0], random_indices_0[i][1]]) # sample seed pixel

    print("seed pixel in selected patch class 0")
    print(selected_patch_0[i][half_patch][half_patch])

    print("seed pixel in data class 1")
    print(data[random_indices_1[i][0], random_indices_1[i][1]]) # sample seed pixel

    print("seed pixel in selected patch class 1")
    print(selected_patch_1[i][half_patch][half_patch])

    return selected_patch_0, selected_patch_1, random_indices_0, random_indices_1