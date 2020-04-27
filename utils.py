def bias_example(distro: list, factor = 2):
    # print("before after")
    # print(distro)
    #hardcore mode:
    sum = 0
    best_index = distro.index(max(distro))

    for i in range(len(distro)):
        distro[i] = 0
    distro[best_index] = 1
    
    # sum = 0
    # for i in range(len(distro)):
    #     distro[i] = distro[i]**factor
    #     sum += distro[i]
    # for i in range(len(distro)):
    #     distro[i] = distro[i]/sum
    # # print(distro)
    
    return distro

