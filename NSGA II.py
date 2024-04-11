import random
import matplotlib.pyplot as plt


# 目标函数 1
def function1(x):
    value = -x**2
    return value


# 目标函数 2
def function2(x):
    value = -(x-2)**2
    return value


# 根据对应目标函数值对列表进行排序
def sort_by_obj_func_value(list1, values, descend_order=False):
    values_in_list1 = [values[i] for i in list1]
    values_list = zip(values_in_list1, list1)
    sorted_values_list = sorted(values_list, reverse=descend_order)   # reverse=False代表升序排列
    sorted_values, sorted_list = zip(*sorted_values_list)
    return list(sorted_list)


# 根据对应聚集度对列表进行排序（聚集度以字典形式存储，键为点索引，值为点的聚集度）
def sort_by_crowding_distance(list1, values, descend_order=False):
    values_in_list1 = [values.get(i) for i in list1]
    values_list = zip(values_in_list1, list1)
    sorted_values_list = sorted(values_list, reverse=descend_order)  # reverse=False代表升序排列
    sorted_values, sorted_list = zip(*sorted_values_list)
    return list(sorted_list)


# 快速非支配排序（返回种群中变量的位置索引，根据该索引可以找到对应目标函数值）
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if ((values1[p] > values1[q] and values2[p] > values2[q])
                    or (values1[p] >= values1[q] and values2[p] > values2[q])
                    or (values1[p] > values1[q] and values2[p] >= values2[q])):
                if q not in S[p]:
                    S[p].append(q)
            elif ((values1[q] > values1[p] and values2[q] > values2[p])
                  or (values1[q] >= values1[p] and values2[q] > values2[p])
                  or (values1[q] > values1[p] and values2[q] >= values2[p])):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while front[i]:
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if n[q] == 0:
                    rank[q] = i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)

    del front[len(front)-1]
    return front


# 计算聚集距离（度）
def calc_crowding_distance(values1, values2, front):
    # 创建聚集度列表，存储多对元组，每个元组为 (该点索引，该点对应聚集度)
    distance = [(0, 0) for i in range(0, len(front))]
    sorted_front = sort_by_obj_func_value(front, values1[:], descend_order=False)
    # 排序后，front点的顺序改变，因此需要将每个点的索引与聚集度对应起来
    distance[0] = (sorted_front[0], 888888888888)
    distance[len(front) - 1] = (sorted_front[len(front) - 1], 888888888888)
    for k in range(1, len(front)-1):
        d1 = abs(values1[sorted_front[k+1]] - values1[sorted_front[k-1]]) / (max(values1)-min(values1))
        d2 = abs(values2[sorted_front[k+1]] - values2[sorted_front[k-1]]) / (max(values2)-min(values2))
        d = d1 + d2
        distance[k] = (sorted_front[k], d)
    # 将元组列表形式转换为字典形式，并返回
    return dict(distance)


# 交叉
def crossover(a, b):
    r = random.random()
    if r > 0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)


# 变异
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = min_x+(max_x-min_x)*random.random()
    return solution


# NSGA2算法整体运行流程
def NSGA2_run(population_N, max_gen_iter, min_x, max_x):
    # 初始种群（包含变量的数值）
    solution = [min_x + (max_x - min_x) * random.random() for i in range(0, population_N)]

    # 遗传迭代
    gen_iter_num = 0
    while gen_iter_num < max_gen_iter:
        print("已进化至第{}代".format(gen_iter_num))
        # 父代选择交叉变异后生成子代，然后将子代与父代合并为 2N的新种群
        solution2N = solution[:]
        while len(solution2N) != 2 * population_N:
            a1 = random.randint(0, population_N - 1)
            b1 = random.randint(0, population_N - 1)
            solution2N.append(crossover(solution[a1], solution[b1]))

        # 将包含父子的 2N规模种群进行排序淘汰，保留规模为 N的下一代种群
        function1_values2 = [function1(solution2N[i]) for i in range(0, 2 * population_N)]
        function2_values2 = [function2(solution2N[i]) for i in range(0, 2 * population_N)]
        ns_index_of_solution2N = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        # 计算 2N规模种群的聚集度情况
        crowding_distance_2N = []
        for i in range(0, len(ns_index_of_solution2N)):
            crowding_distance_2N.append(calc_crowding_distance(function1_values2[:], function2_values2[:],
                                                               ns_index_of_solution2N[i][:]))
        # 根据非支配和聚集度排序将 2N规模的种群缩减至 N
        solutionN = []
        for i in range(0, len(ns_index_of_solution2N)):  # i为每个 Pareto前沿
            # 根据聚集度对单个 Pareto前沿中的点进行排序
            sorted_index_of_front = sort_by_crowding_distance(ns_index_of_solution2N[i][:],
                                                              crowding_distance_2N[i], descend_order=True)
            # 按照综合排序结果，逐个添加，直至规模达到 N
            for index in sorted_index_of_front:
                solutionN.append(solution2N[index])
                if len(solutionN) == population_N:
                    break
            if len(solutionN) == population_N:
                break
        # 更新下一轮的种群
        solution = solutionN
        # 更新迭代次数
        gen_iter_num = gen_iter_num + 1

    return solution


if __name__ == "__main__":
    # 种群规模和迭代次数
    population_N = 30   # 种群规模
    max_gen_iter = 2000   # 最大迭代次数

    # 约束条件
    min_x = -55
    max_x = 55

    # 求解
    solution = NSGA2_run(population_N, max_gen_iter, min_x, max_x)

    # 输出最终解，以及每个解对应的目标函数值
    for index, value in enumerate(solution):
        print("Pareto前沿解{}为: ({}, {}, {})".format(index, value, -1*function1(value), -1*function2(value)))

    # 绘制 Pareto 前沿
    function1_values = [-1 * function1(solution[i]) for i in range(0, population_N)]
    function2_values = [-1 * function2(solution[i]) for i in range(0, population_N)]
    plt.xlabel('Function1', fontsize=15)
    plt.ylabel('Function2', fontsize=15)
    plt.scatter(function1_values, function2_values)
    plt.show()
