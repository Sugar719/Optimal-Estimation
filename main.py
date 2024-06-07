
import numpy as np 
import turtle
import argparse
import time
import matplotlib.pyplot as plt

from maze import Maze, Particle, Robot, WeightedDistribution, weight_gaussian_kernel

plt.rcParams['font.sans-serif']=['Songti SC'] # 用来正常显示中文标签

def main(window_width, window_height, num_particles, sensor_limit_ratio, grid_height, grid_width, num_rows, num_cols, wall_prob, random_seed, robot_speed, kernel_sigma, particle_show_frequency):

    sensor_limit = sensor_limit_ratio * max(grid_height * num_rows, grid_width * num_cols)

    window = turtle.Screen()
    window.setup (width = window_width, height = window_height)

    world = Maze(grid_height = grid_height, grid_width = grid_width, num_rows = num_rows, num_cols = num_cols, wall_prob = wall_prob, random_seed = random_seed)

    x = np.random.uniform(0, world.width)
    y = np.random.uniform(0, world.height)
    bob = Robot(x = x, y = y, maze = world, speed = robot_speed, sensor_limit = sensor_limit)

    particles = list()
    for i in range(num_particles):
        x = np.random.uniform(0, world.width)
        y = np.random.uniform(0, world.height)
        particles.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))

    time.sleep(1)
    world.show_maze()

    # 保存位置信息
    real_positions = []
    estimated_positions = []
    # distances = []
    # plt.ion()
    # fig, ax = plt.subplots()
    # line, = ax.plot(distances, label='距离')
    # plt.xlim([0, 100])
    # plt.legend()
    # plt.show()

    while True:

        # # 保存位置信息
        real_positions.append((bob.x, bob.y))
        estimated_positions.append((np.mean([particle.x for particle in particles]), np.mean([particle.y for particle in particles])))
        # distance = np.sqrt((bob.x - np.mean([particle.x for particle in particles]))**2 + (bob.y - np.mean([particle.y for particle in particles]))**2)
        # distances.append(distance)
        # 更新图形
        # line.set_ydata(distances)  # 更新y轴的数据
        # line.set_xdata(range(len(distances)))  # 更新x轴的数据
        # ax.relim()  # 重新计算坐标轴的限制
        # ax.autoscale_view()  # 自动调整坐标轴的范围
        # plt.draw()  # 更新figure
        # plt.pause(0.01)  # 暂停一会，让GUI有机会更新figure
        # print(distance)

        readings_robot = bob.read_sensor(maze = world)

        particle_weight_total = 0
        for particle in particles:
            readings_particle = particle.read_sensor(maze = world)
            particle.weight = weight_gaussian_kernel(x1 = readings_robot, x2 = readings_particle, std = kernel_sigma)
            particle_weight_total += particle.weight

        world.show_particles(particles = particles, show_frequency = particle_show_frequency)
        world.show_robot(robot = bob)
        world.show_estimated_location(particles = particles)
        world.show_track(positions=real_positions, color='blue') # 绘制轨迹
        world.show_track(positions=estimated_positions, color='red') # 绘制轨迹
        # world.clear_objects()


        # 不能让粒子的权重全为0
        if particle_weight_total == 0:
            particle_weight_total = 1e-8

        # 标准化权重
        for particle in particles:
            particle.weight /= particle_weight_total

        # 重采样
        distribution = WeightedDistribution(particles = particles)
        particles_new = list()

        for i in range(num_particles):

            particle = distribution.random_select()

            if particle is None:
                x = np.random.uniform(0, world.width)
                y = np.random.uniform(0, world.height)
                particles_new.append(Particle(x = x, y = y, maze = world, sensor_limit = sensor_limit))

            else:
                particles_new.append(Particle(x = particle.x, y = particle.y, maze = world, heading = particle.heading, sensor_limit = sensor_limit, noisy = True))

        particles = particles_new

        heading_old = bob.heading
        bob.move(maze = world)
        heading_new = bob.heading
        dh = heading_new - heading_old

        for particle in particles:
            particle.heading = (particle.heading + dh) % 360
            particle.try_move(maze = world, speed = bob.speed)




if __name__ == '__main__':

    window_width = 500  # 窗口宽度
    window_height = 500  # 窗口高度
    num_particles = 500  # 粒子数量
    sensor_limit_ratio = 0.3  # 传感器范围
    grid_height = 100   # 格子高度
    grid_width = 100    # 格子宽度
    num_rows = 25   # 行数
    num_cols = 25   # 列数
    wall_prob = 0.25    # 墙的概率
    random_seed = 200   # 随机种子
    robot_speed = 50    # 机器人速度
    kernel_sigma = 80  # 核函数标准差
    particle_show_frequency = 10    # 粒子显示频率


    main(window_width = window_width, window_height = window_height, num_particles = num_particles, sensor_limit_ratio = sensor_limit_ratio, grid_height = grid_height, grid_width = grid_width, num_rows = num_rows, num_cols = num_cols, wall_prob = wall_prob, random_seed = random_seed, robot_speed = robot_speed, kernel_sigma = kernel_sigma, particle_show_frequency = particle_show_frequency)
