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

    indices_0 = random.sample(zero_indices, k=num_per_class)
    print(f"creating {num_per_class} Randomly chosen 0 indices:")

    indices_1 = random.sample(one_indices, k=num_per_class)
    print(f"creating {num_per_class} Randomly chosen 1 indices:")

    data = training_hsi.img
    patch = patch_size
    n_bands = 224
    half_patch = patch // 2

    # for class 0
    n = len(indices_0)
    n2 = len(indices_1)

    selected_patch_0=np.zeros((n, patch, patch, n_bands))
    selected_patch_1=np.zeros((n2, patch, patch, n_bands))

   

    matrix=zeroPadding.zeroPadding_3D(data,half_patch) #add 0 in every side of the data
    # print(data.shape)
    # print(matrix.shape)

    print(f"indices 0 used: {indices_0}")
    print(f"indices 1 used: {indices_1}")

    for i in range (n): #if padded the index are changing
        # x_pos = indices_0[i][0] - half_patch
        # y_pos = indices_0[i][1] - half_patch
        x_pos = indices_0[i][0]
        y_pos = indices_0[i][1] 
        selected_rows = matrix[range(x_pos,x_pos+2*half_patch+1), :]
        selected_patch_0[i] = selected_rows[:, range(y_pos, y_pos+2*half_patch+1)]

        # print(x_pos, y_pos, indices_0[i][0], indices_0[i][1])

    for i in range (n): #if padded the index are changing
        # x_pos = indices_0[i][0] - half_patch
        # y_pos = indices_0[i][1] - half_patch
        x_pos = indices_1[i][0]
        y_pos = indices_1[i][1] 
        selected_rows = matrix[range(x_pos,x_pos+2*half_patch+1), :]
        selected_patch_1[i] = selected_rows[:, range(y_pos, y_pos+2*half_patch+1)]

        # print(x_pos, y_pos, indices_1[i][0], indices_1[i][1])

    i = 0
    

    # print("seed pixel in data class 0")
    # print(data[indices_0[i][0], indices_0[i][1]]) # sample seed pixel

    # print("seed pixel in selected patch class 0")
    # print(selected_patch_0[i][half_patch][half_patch])

    # print("seed pixel in data class 1")
    # print(data[indices_1[i][0], indices_1[i][1]]) # sample seed pixel

    # print("seed pixel in selected patch class 1")
    # print(selected_patch_1[i][half_patch][half_patch])

    return selected_patch_0, selected_patch_1, indices_0, indices_1

def getSample(hsi, patch_size, num_per_class, indices_0, indices_1):
    training_hsi = hsi
    training_hsi_gt = training_hsi.gt
    print("hsi shape")
    print(training_hsi.img.shape)

    # check if indices is equal gt
    for indice_0 in indices_0:
        if training_hsi_gt[indice_0[0]][indice_0[1]] != 0:
            print("indices 0 does not point to label 0")
            return
        
    for indice_1 in indices_1:
        if training_hsi_gt[indice_1[0]][indice_1[1]] != 1:
            print("indices 1 does not point to label 1")
            return

    
    data = training_hsi.img
    patch = patch_size
    n_bands = 224
    half_patch = patch // 2

    # for class 0
    n = len(indices_0)
    n2 = len(indices_1)

    selected_patch_0=np.zeros((n, patch, patch, n_bands))
    selected_patch_1=np.zeros((n2, patch, patch, n_bands))

   

    matrix=zeroPadding.zeroPadding_3D(data,half_patch) #add 0 in every side of the data
    # print(data.shape)
    # print(matrix.shape)

    print(f"indices 0 used: {indices_0}")
    print(f"indices 1 used: {indices_1}")


    for i in range (n): #if padded the index are changing
        # x_pos = indices_0[i][0] - half_patch
        # y_pos = indices_0[i][1] - half_patch
        x_pos = indices_0[i][0]
        y_pos = indices_0[i][1] 
        selected_rows = matrix[range(x_pos,x_pos+2*half_patch+1), :]
        selected_patch_0[i] = selected_rows[:, range(y_pos, y_pos+2*half_patch+1)]

        # print(x_pos, y_pos, indices_0[i][0], indices_0[i][1])

    for i in range (n): #if padded the index are changing
        # x_pos = indices_0[i][0] - half_patch
        # y_pos = indices_0[i][1] - half_patch
        x_pos = indices_1[i][0]
        y_pos = indices_1[i][1] 
        selected_rows = matrix[range(x_pos,x_pos+2*half_patch+1), :]
        selected_patch_1[i] = selected_rows[:, range(y_pos, y_pos+2*half_patch+1)]

        # print(x_pos, y_pos, indices_1[i][0], indices_1[i][1])

    i = 0
    

    # print("seed pixel in data class 0")
    # print(data[indices_0[i][0], indices_0[i][1]]) # sample seed pixel

    # print("seed pixel in selected patch class 0")
    # print(selected_patch_0[i][half_patch][half_patch])

    # print("seed pixel in data class 1")
    # print(data[indices_1[i][0], indices_1[i][1]]) # sample seed pixel

    # print("seed pixel in selected patch class 1")
    # print(selected_patch_1[i][half_patch][half_patch])

    return selected_patch_0, selected_patch_1