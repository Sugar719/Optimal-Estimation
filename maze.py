import numpy as np
import turtle
import bisect
import argparse


class Maze(object):

    def __init__(self, grid_height, grid_width, maze=None, num_rows=None, num_cols=None, wall_prob=None,
                 random_seed=None):

        self.grid_height = grid_height
        self.grid_width = grid_width

        if maze is not None:
            self.maze = maze
            self.num_rows = maze.shape[0]
            self.num_cols = maze.shape[1]
            self.fix_maze_boundary()
            self.fix_wall_inconsistency()
        else:
            assert num_rows is not None and num_cols is not None and wall_prob is not None, '错误，地图参数设置为0了！！！'
            self.random_maze(num_rows=num_rows, num_cols=num_cols, wall_prob=wall_prob, random_seed=random_seed)

        self.height = self.num_rows * self.grid_height
        self.width = self.num_cols * self.grid_width

        self.turtle_registration()

    def turtle_registration(self):

        turtle.register_shape('tri', ((-3, -2), (0, 3), (3, -2), (0, 0)))

    def check_wall_inconsistency(self):

        wall_errors = list()

        # Check vertical walls
        for i in range(self.num_rows):
            for j in range(self.num_cols - 1):
                if (self.maze[i, j] & 2 != 0) != (self.maze[i, j + 1] & 8 != 0):
                    wall_errors.append(((i, j), 'v'))
        # Check horizonal walls
        for i in range(self.num_rows - 1):
            for j in range(self.num_cols):
                if (self.maze[i, j] & 4 != 0) != (self.maze[i + 1, j] & 1 != 0):
                    wall_errors.append(((i, j), 'h'))

        return wall_errors

    def fix_wall_inconsistency(self, verbose=True):
        wall_errors = self.check_wall_inconsistency()

        if wall_errors and verbose:
            print('迷宫中的墙壁不一致了')

        for (i, j), error in wall_errors:
            if error == 'v':
                self.maze[i, j] |= 2
                self.maze[i, j + 1] |= 8
            elif error == 'h':
                self.maze[i, j] |= 4
                self.maze[i + 1, j] |= 1
            else:
                raise Exception('未知的墙体紊乱问题')
        return

    def fix_maze_boundary(self):
        '''
        确定迷宫的边界
        '''
        for i in range(self.num_rows):
            self.maze[i, 0] |= 8
            self.maze[i, -1] |= 2
        for j in range(self.num_cols):
            self.maze[0, j] |= 1
            self.maze[-1, j] |= 4

    def random_maze(self, num_rows, num_cols, wall_prob, random_seed=None):

        if random_seed is not None:
            np.random.seed(random_seed)
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.maze = np.zeros((num_rows, num_cols), dtype=np.int8)
        for i in range(self.num_rows):
            for j in range(self.num_cols - 1):
                if np.random.rand() < wall_prob:
                    self.maze[i, j] |= 2
        for i in range(self.num_rows - 1):
            for j in range(self.num_cols):
                if np.random.rand() < wall_prob:
                    self.maze[i, j] |= 4

        self.fix_maze_boundary()
        self.fix_wall_inconsistency(verbose=False)

    def permissibilities(self, cell):
        '''
        确定一个格子的四个方向是否有墙
        '''
        cell_value = self.maze[cell[0], cell[1]]
        return (cell_value & 1 == 0, cell_value & 2 == 0, cell_value & 4 == 0, cell_value & 8 == 0)

    def distance_to_walls(self, coordinates):
        '''
        确定一个坐标点到四个方向的距离
        '''

        x, y = coordinates

        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
        d1 = y - y // self.grid_height * self.grid_height
        while self.permissibilities(cell=(i, j))[0]:
            i -= 1
            d1 += self.grid_height

        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
        d2 = self.grid_width - (x - x // self.grid_width * self.grid_width)
        while self.permissibilities(cell=(i, j))[1]:
            j += 1
            d2 += self.grid_width

        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
        d3 = self.grid_height - (y - y // self.grid_height * self.grid_height)
        while self.permissibilities(cell=(i, j))[2]:
            i += 1
            d3 += self.grid_height

        i = int(y // self.grid_height)
        j = int(x // self.grid_width)
        d4 = x - x // self.grid_width * self.grid_width
        while self.permissibilities(cell=(i, j))[3]:
            j -= 1
            d4 += self.grid_width

        return [d1, d2, d3, d4]

    def show_maze(self):

        turtle.setworldcoordinates(0, 0, self.width * 1.005, self.height * 1.005)

        wally = turtle.Turtle()
        wally.speed(0)
        wally.width(1.5)
        wally.hideturtle()
        turtle.tracer(0, 0)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                permissibilities = self.permissibilities(cell=(i, j))
                turtle.up()
                wally.setposition((j * self.grid_width, i * self.grid_height))
                # 设置方向
                # 0 - 东, 90 - 北, 180 - 西, 270 - 南
                wally.setheading(0)
                if not permissibilities[0]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_width)
                wally.setheading(90)
                wally.up()
                if not permissibilities[1]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_height)
                wally.setheading(180)
                wally.up()
                if not permissibilities[2]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_width)
                wally.setheading(270)
                wally.up()
                if not permissibilities[3]:
                    wally.down()
                else:
                    wally.up()
                wally.forward(self.grid_height)
                wally.up()

        turtle.update()

    def weight_to_color(self, weight):
        # 设置颜色
        return '#%02x00%02x' % (int(weight * 255), int((1 - weight) * 255))

    def show_particles(self, particles, show_frequency=10):

        turtle.shape('tri')

        for i, particle in enumerate(particles):
            if i % show_frequency == 0:
                turtle.setposition((particle.x, particle.y))
                turtle.setheading(90 - particle.heading)
                turtle.color(self.weight_to_color(particle.weight))
                turtle.stamp()

        turtle.update()

    def show_estimated_location(self, particles):
        '''
        显示估计位置
        '''

        x_accum = 0
        y_accum = 0
        heading_accum = 0
        weight_accum = 0

        num_particles = len(particles)

        for particle in particles:
            weight_accum += particle.weight
            x_accum += particle.x * particle.weight
            y_accum += particle.y * particle.weight
            heading_accum += particle.heading * particle.weight

        if weight_accum == 0:
            return False

        x_estimate = x_accum / weight_accum
        y_estimate = y_accum / weight_accum
        heading_estimate = heading_accum / weight_accum

        turtle.color('orange')
        turtle.setposition(x_estimate, y_estimate)
        turtle.setheading(90 - heading_estimate)
        turtle.shape('turtle')
        turtle.stamp()
        turtle.update()

    def show_robot(self, robot):

        turtle.color('green')
        turtle.shape('turtle')
        turtle.shapesize(0.7, 0.7)
        turtle.setposition((robot.x, robot.y))
        turtle.setheading(90 - robot.heading)
        turtle.stamp()
        turtle.update()

    def show_track(self, positions, color='red'):

        turtle.color(color)
        turtle.shape('circle')
        for position in positions:
            turtle.setposition(position)
            turtle.stamp()
        turtle.update()

    def clear_objects(self):
        turtle.clearstamps()


class Particle(object):

    def __init__(self, x, y, maze, heading=None, weight=1.0, sensor_limit=None, noisy=False):

        if heading is None:
            heading = np.random.uniform(0, 360)

        self.x = x
        self.y = y
        self.heading = heading
        self.weight = weight
        self.maze = maze
        self.sensor_limit = sensor_limit

        if noisy:
            std = max(self.maze.grid_height, self.maze.grid_width) * 0.2
            self.x = self.add_noise(x=self.x, std=std)
            self.y = self.add_noise(x=self.y, std=std)
            self.heading = self.add_noise(x=self.heading, std=360 * 0.05)

        self.fix_invalid_particles()

    def fix_invalid_particles(self):

        # 修复粒子的位置和朝向
        if self.x < 0:
            self.x = 0
        if self.x > self.maze.width:
            self.x = self.maze.width * 0.9999
        if self.y < 0:
            self.y = 0
        if self.y > self.maze.height:
            self.y = self.maze.height * 0.9999
        '''
        if self.heading > 360:
            self.heading -= 360
        if self.heading < 0:
            self.heading += 360
        '''
        self.heading = self.heading % 360

    @property
    def state(self):

        return (self.x, self.y, self.heading)

    def add_noise(self, x, std):

        return x + np.random.normal(0, std)

    def read_sensor(self, maze):

        readings = maze.distance_to_walls(coordinates=(self.x, self.y))

        heading = self.heading % 360

        # 去除朝向信息
        if heading >= 45 and heading < 135:
            readings = readings
        elif heading >= 135 and heading < 225:
            readings = readings[-1:] + readings[:-1]
            #readings = [readings[3], readings[0], readings[1], readings[2]]
        elif heading >= 225 and heading < 315:
            readings = readings[-2:] + readings[:-2]
            #readings = [readings[2], readings[3], readings[0], readings[1]]
        else:
            readings = readings[-3:] + readings[:-3]
            #readings = [readings[1], readings[2], readings[3], readings[0]]

        if self.sensor_limit is not None:
            for i in range(len(readings)):
                if readings[i] > self.sensor_limit:
                    readings[i] = self.sensor_limit

        return readings

    def try_move(self, speed, maze, noisy=False):

        heading = self.heading
        heading_rad = np.radians(heading)

        dx = np.sin(heading_rad) * speed
        dy = np.cos(heading_rad) * speed

        x = self.x + dx
        y = self.y + dy

        gj1 = int(self.x // maze.grid_width)
        gi1 = int(self.y // maze.grid_height)
        gj2 = int(x // maze.grid_width)
        gi2 = int(y // maze.grid_height)

        # 检查是否碰到墙壁
        if gi2 < 0 or gi2 >= maze.num_rows or gj2 < 0 or gj2 >= maze.num_cols:
            return False

        # 移动到同一个格子
        if gi1 == gi2 and gj1 == gj2:
            self.x = x
            self.y = y
            return True
        # 移动到相邻的格子
        elif abs(gi1 - gi2) == 1 and abs(gj1 - gj2) == 0:
            if maze.maze[min(gi1, gi2), gj1] & 4 != 0:
                return False
            else:
                self.x = x
                self.y = y
                return True
        # 移动到相邻的格子
        elif abs(gi1 - gi2) == 0 and abs(gj1 - gj2) == 1:
            if maze.maze[gi1, min(gj1, gj2)] & 2 != 0:
                return False
            else:
                self.x = x
                self.y = y
                return True
        # 移动到对角的格子
        elif abs(gi1 - gi2) == 1 and abs(gj1 - gj2) == 1:

            x0 = max(gj1, gj2) * maze.grid_width
            y0 = (y - self.y) / (x - self.x) * (x0 - self.x) + self.y

            if maze.maze[int(y0 // maze.grid_height), min(gj1, gj2)] & 2 != 0:
                return False

            y0 = max(gi1, gi2) * maze.grid_height
            x0 = (x - self.x) / (y - self.y) * (y0 - self.y) + self.x

            if maze.maze[min(gi1, gi2), int(x0 // maze.grid_width)] & 4 != 0:
                return False

            self.x = x
            self.y = y
            return True

        else:
            raise Exception('错误，粒子移动到了不相邻的格子上')


class Robot(Particle):

    def __init__(self, x, y, maze, heading=None, speed=1.0, sensor_limit=None, noisy=True):

        super(Robot, self).__init__(x=x, y=y, maze=maze, heading=heading, sensor_limit=sensor_limit, noisy=noisy)
        self.step_count = 0
        self.noisy = noisy
        self.time_step = 0
        self.speed = speed

    def choose_random_direction(self):

        self.heading = np.random.uniform(0, 360)

    def add_sensor_noise(self, x, z=0.05):

        readings = list(x)

        for i in range(len(readings)):
            std = readings[i] * z / 2
            readings[i] = readings[i] + np.random.normal(0, std)

        return readings

    def read_sensor(self, maze):

        # 读取传感器数据

        readings = super(Robot, self).read_sensor(maze=maze)
        if self.noisy == True:
            readings = self.add_sensor_noise(x=readings)

        return readings

    def move(self, maze):

        while True:
            self.time_step += 1
            if self.try_move(speed=self.speed, maze=maze, noisy=False):
                break
            self.choose_random_direction()


class WeightedDistribution(object):

    def __init__(self, particles):

        accum = 0.0
        self.particles = particles
        self.distribution = list()
        for particle in self.particles:
            accum += particle.weight
            self.distribution.append(accum)

    def random_select(self):

        try:
            return self.particles[bisect.bisect_left(self.distribution, np.random.uniform(0, 1))]
        except IndexError:
            # 当粒子权重全为0时，返回None
            return None


def euclidean_distance(x1, x2):
    return np.linalg.norm(np.asarray(x1) - np.asarray(x2))


def weight_gaussian_kernel(x1, x2, std=10):
    distance = euclidean_distance(x1=x1, x2=x2)
    return np.exp(-distance ** 2 / (2 * std))
