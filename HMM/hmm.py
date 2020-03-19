"""
    @author:CodingMarathon
    @date:2020-03-18
    @blog:https://blog.csdn.net/Elenstone/article/details/104902120
"""
from typing import List, Any

import numpy as np
import random


class HMM(object):
    def __init__(self, n, m, a=None, b=None, pi=None):
        # 可能的隐藏状态数
        self.N = n
        # 可能的观测数
        self.M = m
        # 状态转移概率矩阵
        self.A = a
        # 观测概率矩阵
        self.B = b
        # 初始状态概率矩阵
        self.Pi = pi
        # 观测序列
        self.X = None
        # 状态序列
        self.Y = None
        # 序列长度
        self.T = 0

        # 定义前向算法
        self.alpha = None
        # 定义后向算法
        self.beta = None

    def forward(self, x):
        """
        前向算法
        计算给定模型参数和观测序列的情况下，观测序列出现的最大概率
        :param x: 观测序列
        :return: 观测值
        """
        # 序列长度
        self.T = len(x)
        self.X = np.array(x)

        # alpha是一个具有T行N列的矩阵
        self.alpha = np.zeros((self.T, self.N))

        # 初始状态
        for i in range(self.N):
            self.alpha[0][i] = self.Pi[i] * self.B[i][self.X[0]]

        # 递推
        for t in range(1, self.T):
            for i in range(self.N):
                probability_sum = 0
                for j in range(self.N):
                    probability_sum += self.alpha[t - 1][j] * self.A[j][i]
                self.alpha[t][i] = probability_sum * self.B[i][self.X[t]]
        # 终止
        return sum(self.alpha[self.T - 1])

    def backward(self, x):
        """
        后向算法
        """
        # 序列长度
        self.T = len(x)
        self.X = np.array(x)

        # beta是一个T行N列的矩阵
        self.beta = np.zeros((self.T, self.N))

        # 当t=T时，置值为1
        for i in range(self.N):
            self.beta[self.T - 1][i] = 1

        # 从t=T-1递推到t=1
        for t in range(self.T - 2, -1, -1):
            for i in range(self.N):
                for j in range(self.N):
                    self.beta[t][i] += self.A[i][j] * self.B[j][self.X[t + 1]] * self.beta[t + 1][j]

        # 终止
        sum_probability = 0
        for i in range(self.N):
            sum_probability += self.Pi[i] * self.B[i][self.X[0]] * self.beta[0][i]

        return sum_probability

    def calculate_gamma(self, t, i):
        """
        给定模型参数和观测序列，计算在t时刻处于状态q_i的概率
        :param i: 状态
        :param t: 时刻
        :return: 时刻t处于状态i的概率
        """
        # 分子
        numerator = self.alpha[t][i] * self.beta[t][i]
        # 分母
        denominator = 0
        for j in range(self.N):
            denominator += self.alpha[t][j] * self.beta[t][j]

        return numerator / denominator

    def calculate_xi(self, t, i, j):
        """
        给定模型参数和观测序列，在时刻t处于状态q_i且时刻t+1处于状态q_j的概率
        :param i: 时刻t的状态
        :param j: 时刻t+1的状态
        :param t: 时刻t
        :return: 在时刻t处于状态q_i且时刻t+1处于状态q_j的概率
        """
        # 分子
        numerator = self.alpha[t][i] * self.A[i][j] * self.B[j][self.X[t + 1]] * self.beta[t + 1][j]
        # 分母
        denominator = 0
        for i in range(self.N):
            for j in range(self.N):
                denominator += self.alpha[t][i] * self.A[i][j] * self.B[j][self.X[t + 1]] * self.beta[t + 1][j]

        return numerator / denominator

    def init(self):
        """
        训练时初始化HMM模型
        """
        self.A = np.zeros((self.N, self.N))
        self.B = np.zeros((self.N, self.M))
        self.Pi = np.zeros(self.N)

    def train(self, train_data):
        """
        训练模型，使用最大似然估计
        :param train_data: 训练数据，每一个样本：[观测值，隐藏状态值]
        :return: 返回一个HMM模型
        """

        self.T = int(len(train_data[0]) / 2)
        sample_num = len(train_data)

        # 初始化
        self.init()

        # 初始状态概率矩阵的估计
        for sequence in train_data:
            self.Pi[sequence[0 + self.T]] += 1
        self.Pi = self.Pi / sample_num

        # 状态转移矩阵的估计
        a_num = np.zeros((self.N, self.N))
        for sequence in train_data:
            for i in range(self.T - 1):
                a_num[sequence[i + self.T]][sequence[i + 1 + self.T]] += 1.0
        temp = a_num.sum(axis=1).reshape((3, 1))
        self.A = a_num / temp

        # 发射概率矩阵的估计
        b_num = np.zeros((self.N, self.M))
        for sequence in train_data:
            for i in range(self.T - 1):
                b_num[sequence[i + self.T]][sequence[i]] += 1.0
        temp = b_num.sum(axis=1).reshape((3, 1))
        self.B = b_num / temp

    def baum_welch(self, x, criterion=0.001):
        self.T = len(x)
        self.X = x

        while True:
            # 为了得到alpha和beta的矩阵
            _ = self.forward(self.X)
            _ = self.backward(self.X)
            xi = np.zeros((self.T - 1, self.N, self.N), dtype=float)
            for t in range(self.T - 1):
            # 笨办法
            # for i in range(self.N):
            # gamma[t][i] = self.calculate_gamma(t, i)
            #         for j in range(self.N):
            #             xi[t][i][j] = self.calculate_psi(t, i, j)
            # for i in range(self.N):
            #     gamma[self.T-1][i] = self.calculate_gamma(self.T-1, i)

            # numpy矩阵的办法
                denominator = np.sum(np.dot(self.alpha[t, :], self.A) *
                                     self.B[:, self.X[t + 1]] * self.beta[t + 1, :])
                for i in range(self.N):
                    molecular = self.alpha[t, i] * self.A[i, :] * self.B[:, self.X[t+1]]*self.beta[t+1, :]
                    xi[t, i, :] = molecular / denominator
            gamma = np.sum(xi, axis=2)
            prod = (self.alpha[self.T-1, :]*self.beta[self.T-1, :])
            gamma = np.vstack((gamma, prod / np.sum(prod)))

            new_pi = gamma[0, :]
            new_a = np.sum(xi, axis=0) / np.sum(gamma[:-1, :], axis=0).reshape(-1, 1)
            new_b = np.zeros(self.B.shape, dtype=float)

            for k in range(self.B.shape[1]):
                mask = self.X == k
                new_b[:, k] = np.sum(gamma[mask, :], axis=0) / np.sum(gamma, axis=0)

            if np.max(abs(self.Pi - new_pi)) < criterion and \
                    np.max(abs(self.A - new_a)) < criterion and \
                    np.max(abs(self.B - new_b)) < criterion:
                break

        self.A, self.B, self.Pi = new_a, new_b, new_pi

    def viterbi(self, x):
        self.X = x
        self.T = len(x)
        self.Y = np.zeros(self.N)
        # 初始化delta和xi
        delta = np.zeros((self.T, self.N))
        psi = np.zeros((self.T, self.N))

        # 初始化，t=1时
        for i in range(self.N):
            delta[0][i] = self.Pi[i] * self.B[i][self.X[0]]
            psi[0][i] = 0

        # 递推
        for t in range(1, self.T):
            for i in range(self.N):
                temp = 0
                index = 0
                for j in range(self.N):
                    if temp < delta[t - 1][j] * self.A[j][i]:
                        temp = delta[t - 1][j] * self.A[j][i]
                        index = j
                delta[t][i] = temp * self.B[i][self.X[t]]
                psi[t][i] = j

        # 最终
        self.Y[-1] = delta.argmax(axis=1)[-1]
        p = delta[self.T - 1][int(self.Y[-1])]

        # 回溯
        for i in range(self.T - 1, 0, -1):
            self.Y[i - 1] = psi[i][int(self.Y[i])]

        return p, self.Y


def generate_train_data(n, m, t, a, b, pi, nums=10000):
    """
    生成训练数据
    :param pi: 初始状态概率矩阵
    :param b: 发射概率矩阵
    :param a: 状态转移矩阵
    :param n: 隐藏状态数量
    :param m:观测值数量
    :param t: 序列长度
    :param nums: 样本数量
    :return: 训练数据集
    """
    train_data = []

    for i in range(nums):
        state_sequence = []
        observation_sequence = []
        # 初始状态
        temp = 0
        p = random.random()
        for j in range(n):
            temp += pi[j]
            if p > temp:
                continue
            else:
                state_sequence.append(j)
                break

        # 递推
        for t_index in range(t):
            # 生成状态
            if t_index != 0:
                temp = 0
                p = random.random()
                for state in range(n):
                    temp += a[state_sequence[-1]][state]
                    if p > temp:
                        continue
                    else:
                        state_sequence.append(state)
                        break
            # 生成观测序列
            temp = 0
            p = random.random()
            for observation in range(m):
                temp += b[state_sequence[-1]][observation]
                if p > temp:
                    continue
                else:
                    observation_sequence.append(observation)
                    break
        observation_sequence.extend(state_sequence)
        train_data.append(observation_sequence)
    return train_data


if __name__ == '__main__':
    q = [1, 2, 3]
    v = ["红", "白"]
    n = len(q)
    m = len(v)
    x = ["红", "白", "红"]
    # 建立一个字典
    char_to_id = {}
    id_to_char = {}

    for i in v:
        char_to_id[i] = len(char_to_id)
        id_to_char[char_to_id[i]] = i

    X = [char_to_id[i] for i in x]

    a = np.array([[0.5, 0.2, 0.3],
                  [0.3, 0.5, 0.2],
                  [0.2, 0.3, 0.5]])
    b = np.array([[0.5, 0.5],
                  [0.4, 0.6],
                  [0.7, 0.3]])
    pi = np.array([0.2, 0.4, 0.4])

    # hmm = HMM(n, m, a, b, pi)
    # print(hmm.backward(X))

    # 预测
    # 产生数据
    # train_data = generate_train_data(n, m, 8, a, b, pi)
    # print(train_data)
    # hmm = HMM(n, m)
    # hmm.train(train_data)
    # print(hmm.Pi)
    # print(hmm.A)
    # print(hmm.B)

    # # 使用维特比
    # hmm = HMM(n, m, a, b, pi)
    # p, y = hmm.viterbi(X)
    # print(p)
    # print(y)

    # 使用baum-welch
    hmm = HMM(n, m, a, b, pi)
    hmm.baum_welch(X)
