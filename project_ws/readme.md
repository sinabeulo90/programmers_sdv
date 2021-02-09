![Do not collide!](https://github.com/sinabeulo90/programmers_sdv/raw/main/project_ws/demo/Do%20not%20collide!.gif)


## Programmers project - Do not collide!

```
(terminal1)
cd src
roslaunch main.launch

(t2 -  spawn the autonomous agent)
rosrun cv_agents spawn_agent.py
```


## Guideline
1. Implement PID speed controller
2. Implement pure pursuit (or stanley, PID, MPC) as a lateral controller


### Stanley Method

```python
    ...

    target_speed = 20.0 / 3.6

    while not rospy.is_shutdown():
        """
        generate acceleration ai, and steering di
        - target_speed: 자차량의 목표 속도
        - state: 자차량의 정보를 담고 있는 객체
        """
        # 목표 속도와 차량의 현재 속도 사이의 오차를 계산하고, 오차의 50% 만큼 차량을 가속
        speed_error = target_speed - state.v
        ai = 0.5 * speed_error

        # 차량의 현재 위치(x, y, yaw)에서 주행하려는 경로(trajectory)를 따라 움직이기 위해
        # Stanley 방법을 사용하여 steering을 계산
        di = stanley_control(state.x, state.y, state.yaw, state.v, traj_x, traj_y, traj_yaw)

        # 위에서 계산된 가속도와 steering 값을 현재 차량에 적용
        state.update(ai, di)

        ...
```


3. Implement optimal frenet planning algorithm

### Optimal Frenet Planning

```python
    ...

    """
    Optimal frenet planning algorithm을 사용하기 위해, 자차량 초기 상태를 정의
    - v: 속도
    - a: 가속도
    - s, d: 자차량의 frenet frame 좌표 위치
    """
    v = state.v
    a = 0
    s, d = get_frenet(state.x, state.y, traj_x, traj_y);

    """
    자차량의 s 방향 초기 상태 및 조건 계산
    - si: 자차량의 s방향
    - si_d: 자차량의 s방향 초기 속도
    - si_dd: 자차량의 s방향 초기 가속도
    - sf_d: 자차량의 s방향 목표 속도
    - sf_dd: 자차량의 s방향 목표 가속도
    """
    si = s
    si_d = v * np.cos(state.yaw)
    si_dd = a * np.cos(state.yaw)
    sf_d = target_speed
    sf_dd = 0

    """
    자차량의 d 방향 초기 상태 및 조건 계산
    - di: 자차량의 d방향
    - di_d: 자차량의 d방향 초기 속도
    - si_dd: 자차량의 d방향 초기 가속도
    - df_d: 자차량의 d방향 목표 속도
    - df_dd: 자차량의 d방향 목표 가속도
    - opt_d: planning 된 경로 중 최적 경로의 d방향 값
    """
    di = d
    di_d = v * np.sin(state.yaw)
    di_dd = a * np.sin(state.yaw)
    df_d = 0
    df_dd = 0
    opt_d = 0

    while not rospy.is_shutdown():
        """
        optimal planning 수행
        - path: planning 된 모든 경로
        - opt_ind: 모든 경로 중 최적 경로의 index
        """
        path, opt_ind = frenet_optimal_planning(si, si_d, si_dd, sf_d, sf_dd,
                                                di, di_d, di_dd, df_d, df_dd,
                                                obs, traj_x, traj_y, traj_yaw, opt_d)
        
        """
        최적 경로에서의 속도를 현재 차량의 목표 속도로 사용
        - s_d: 최적 경로의 s방향의 속도
        - d_d: 최적 경로의 d방향의 속도
        """
        s_d = path[opt_ind].s_d[0]
        d_d = path[opt_ind].d_d[0]
        target_speed = np.sqrt(s_d**2 + d_d**2)

        # 목표 속도와 차량의 현재 속도 사이의 오차를 계산하고, 오차의 50% 만큼 차량을 가속
        speed_error = target_speed - state.v
        ai = 0.5 * speed_error

        """
        planning으로 찾아낸 경로 중 최적 경로(path[opt_ind])를 따라 움직이기 위해
        Stanley 방법을 사용하여 steering을 계산
        """
        di = stanley_control(state.x, state.y, state.yaw, state.v, path[opt_ind].x, path[opt_ind].y, path[opt_ind].yaw)

        # update state with acc, delta
        state.update(ai, di)

        """
        업데이트 된 자차량의 정보를 기준으로 새로 planning하기 위해 필요 인자들을 업데이트
        """
        si, di = get_frenet(state.x, state.y, traj_x, traj_y)
        si_d = path[opt_ind].s_d[1]
        si_dd = path[opt_ind].s_dd[1]
        di_d = path[opt_ind].d_d[1]
        di_dd = path[opt_ind].d_dd[1]
        # consistency cost를 위해 update
        opt_d = path[opt_ind].d[-1]

        ...
```
