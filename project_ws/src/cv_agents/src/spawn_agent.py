#!/usr/bin/python
#-*- coding: utf-8 -*-

import rospy
import numpy as np
import math
import tf

import rospkg
import sys

from scipy.interpolate import interp1d

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Quaternion
from object_msgs.msg import Object

import pickle
import argparse

from stanley import stanley_control
from optimal_trajectory_Frenet import frenet_optimal_planning, get_frenet

from visualization_msgs.msg import MarkerArray
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point


rospack = rospkg.RosPack()
path = rospack.get_path("map_server")

rn_id = dict()

rn_id[5] = {
    'left': [18, 2, 11, 6, 13, 8, 15, 10, 26, 0]  # ego route
}

def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def interpolate_waypoints(wx, wy, space=0.5):
    _s = 0
    s = [0]
    for i in range(1, len(wx)):
        prev_x = wx[i - 1]
        prev_y = wy[i - 1]
        x = wx[i]
        y = wy[i]

        dx = x - prev_x
        dy = y - prev_y

        _s = np.hypot(dx, dy)
        s.append(s[-1] + _s)

    fx = interp1d(s, wx)
    fy = interp1d(s, wy)
    ss = np.linspace(0, s[-1], num=int(s[-1] / space) + 1, endpoint=True)

    dxds = np.gradient(fx(ss), ss, edge_order=1)
    dyds = np.gradient(fy(ss), ss, edge_order=1)
    wyaw = np.arctan2(dyds, dxds)

    return {
        "x": fx(ss),
        "y": fy(ss),
        "yaw": wyaw,
        "s": ss
    }


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, dt=0.1, WB=2.6):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))
        self.dt = dt
        self.WB = WB

    def update(self, a, delta):
        dt = self.dt
        WB = self.WB

        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v / WB * math.tan(delta) * dt
        self.yaw = pi_2_pi(self.yaw)
        self.v += a * dt
        self.rear_x = self.x - ((WB / 2) * math.cos(self.yaw))
        self.rear_y = self.y - ((WB / 2) * math.sin(self.yaw))

    def calc_distance(self, point_x, point_y):
        dx = self.rear_x - point_x
        dy = self.rear_y - point_y
        return math.hypot(dx, dy)


def get_ros_msg(x, y, yaw, v, id):
    quat = tf.transformations.quaternion_from_euler(0, 0, yaw)

    m = Marker()
    m.header.frame_id = "/map"
    m.header.stamp = rospy.Time.now()
    m.id = id
    m.type = m.CUBE

    m.pose.position.x = x + 1.3 * math.cos(yaw)
    m.pose.position.y = y + 1.3 * math.sin(yaw)
    m.pose.position.z = 0.75
    m.pose.orientation = Quaternion(*quat)

    m.scale.x = 4.475
    m.scale.y = 1.850
    m.scale.z = 1.645

    m.color.r = 93 / 255.0
    m.color.g = 122 / 255.0
    m.color.b = 177 / 255.0
    m.color.a = 0.97

    o = Object()
    o.header.frame_id = "/map"
    o.header.stamp = rospy.Time.now()
    o.id = id
    o.classification = o.CLASSIFICATION_CAR
    o.x = x
    o.y = y
    o.yaw = yaw
    o.v = v
    o.L = m.scale.x
    o.W = m.scale.y

    return {
        "object_msg": o,
        "marker_msg": m,
        "quaternion": quat
    }


# 경로 후보들과 최적 경로를 표시하는 메시
def get_traj_msg(paths, opt_ind, traj_x, traj_y, maps):
    ma = MarkerArray()

    for id, path in enumerate(paths):
        m = Marker()
        m.header.frame_id = "/map"
        m.header.stamp = rospy.Time.now()
        m.id = (id + 1) * 10
        m.type = m.POINTS
        m.lifetime.nsecs = 10


        c = ColorRGBA()
        if opt_ind == id:
            m.scale.x = 1.0
            m.scale.y = 1.0
            m.scale.z = 1.0
            c.r = 0 / 255.0
            c.g = 0 / 255.0
            c.b = 255 / 255.0
            c.a = 1
        else:
            m.scale.x = 0.2
            m.scale.y = 0.2
            m.scale.z = 0.2
            c.r = 255 / 255.0
            c.g = 0 / 255.0
            c.b = 0 / 255.0
            c.a = 0.3

        for x, y in zip(path.x, path.y):
            p = Point()
            p.x, p.y = x, y
            m.points.append(p)
            m.colors.append(c)

        ma.markers.append(m)

    return ma

# 주차된 차량 데이터
car2_data = None
car3_data = None

# 주차된 차량에 대한 토픽을 업데이트하기 위한 콜백함수
def car2_callback(data):
    global car2_data
    car2_data = data
def car3_callback(data):
    global car3_data
    car3_data = data

# 주차된 차량의 영역을 좌표로 변환
def get_car_grid(data):
    Ls = np.linspace(-data.L/2, data.L/2, data.L*10)
    Ws = np.linspace(-data.W/2, data.W/2, data.W*10)

    dy, dx = np.meshgrid(Ls, Ws)
    y = data.y + dy*np.sin(data.yaw)
    x = data.x + dx*np.cos(data.yaw)
    return y, x

def get_parking_car_grid():
    car2_y, car2_x = get_car_grid(car2_data)
    car3_y, car3_x = get_car_grid(car3_data)
    return (car2_x, car2_y), (car3_x, car3_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Spawn a CV agent')

    parser.add_argument("--id", "-i", type=int, help="agent id", default=1)
    parser.add_argument("--route", "-r", type=int,
                        help="start index in road network. select in [1, 3, 5, 10]", default=5)
    parser.add_argument("--dir", "-d", type=str, default="left", help="direction to go: [left, straight, right]")
    args = parser.parse_args()

    rospy.init_node("three_cv_agents_node_" + str(args.id))

    id = args.id
    tf_broadcaster = tf.TransformBroadcaster()
    marker_pub = rospy.Publisher("/objects/marker/car_" + str(id), Marker, queue_size=1)
    object_pub = rospy.Publisher("/objects/car_" + str(id), Object, queue_size=1)

    # 경로를 RVIZ에 표시하기 위한 publisher
    traj_pub = rospy.Publisher("/rviz/trajectory", MarkerArray, queue_size=10)

    # 주차된 차량에 대한 토픽을 구독하기 위한 subscriber
    sub2 = rospy.Subscriber("/objects/car_2", Object, car2_callback, queue_size=1)
    sub3 = rospy.Subscriber("/objects/car_3", Object, car3_callback, queue_size=1)

    # 주차된 차량의 정보를 초기 갱신
    while car2_data is None or car3_data is None:
        pass

    start_node_id = args.route
    route_id_list = [start_node_id] + rn_id[start_node_id][args.dir]

    ind = 100

    with open(path + "/src/route.pkl", "rb") as f:
        nodes = pickle.load(f)

    wx = []
    wy = []
    wyaw = []
    for _id in route_id_list:
        wx.append(nodes[_id]["x"][1:])
        wy.append(nodes[_id]["y"][1:])
        wyaw.append(nodes[_id]["yaw"][1:])
    wx = np.concatenate(wx)
    wy = np.concatenate(wy)
    wyaw = np.concatenate(wyaw)

    waypoints = {"x": wx, "y": wy, "yaw": wyaw}

    target_speed = 20.0 / 3.6
    state = State(x=waypoints["x"][ind], y=waypoints["y"][ind], yaw=waypoints["yaw"][ind], v=0.1, dt=0.1)

    traj_x = waypoints["x"]
    traj_y = waypoints["y"]
    traj_yaw = waypoints["yaw"]

    traj_s = np.zeros(traj_x.shape)
    for i in range(len(traj_x) - 1):
        x = traj_x[i]
        y = traj_y[i]
        sd = get_frenet(x, y, traj_x, traj_y)
        traj_yaw[i] = sd[0]

    # 자차량 관련 initial condition
    v = 0.1
    a = 0
    s, d = get_frenet(state.x, state.y, traj_x, traj_y);

    # s 방향 초기조건
    si = s
    si_d = v*np.cos(state.yaw)
    si_dd = a*np.cos(state.yaw)
    sf_d = target_speed
    sf_dd = 0

    # d 방향 초기조건
    di = d
    di_d = v*np.sin(state.yaw)
    di_dd = a*np.sin(state.yaw)
    df_d = 0
    df_dd = 0

    opt_d = 0

    # 장애물 정보
    obs2_xy, obs3_xy = get_parking_car_grid()
    obs = np.array([[obs2_xy[0], obs2_xy[1]], [obs3_xy[0], obs3_xy[1]]])

    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        # optimal planning 수행 (output : valid path & optimal path index)
        path, opt_ind = frenet_optimal_planning(si, si_d, si_dd,
                                                sf_d, sf_dd, di, di_d, di_dd, df_d, df_dd, obs, traj_x, traj_y, traj_yaw, opt_d)

        '''
        다음 시뮬레이션 step 에서 사용할 initial condition update.
        본 파트에서는 planning 만 수행하고 control 은 따로 수행하지 않으므로,
        optimal trajectory 중 현재 위치에서 한개 뒤 index 를 다음 step 의 초기초건으로 사용.
        '''
        si_d = path[opt_ind].s_d[1]
        si_dd = path[opt_ind].s_dd[1]
        di_d = path[opt_ind].d_d[1]
        di_dd = path[opt_ind].d_dd[1]

        # consistency cost를 위해 update
        opt_d = path[opt_ind].d[-1]

        # generate acceleration ai, and steering di
        speed_error = target_speed - state.v
        ai = 0.5 * speed_error
        di = stanley_control(state.x, state.y, state.yaw, state.v, path[opt_ind].x, path[opt_ind].y, path[opt_ind].yaw)

        # update state with acc, delta
        state.update(ai, di)

        si, di = get_frenet(state.x, state.y, traj_x, traj_y)

        # vehicle state --> topic msg
        msg = get_ros_msg(state.x, state.y, state.yaw, state.v, id=id)
        traj_msg = get_traj_msg(path, opt_ind, traj_x, traj_y, traj_s)

        # send tf
        tf_broadcaster.sendTransform(
            (state.x, state.y, 1.5),
            msg["quaternion"],
            rospy.Time.now(),
            "/car_" + str(id), "/map"
        )

        # publish vehicle state in ros msg
        object_pub.publish(msg["object_msg"])
        traj_pub.publish(traj_msg)

        r.sleep()
