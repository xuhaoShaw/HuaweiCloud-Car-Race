# ROS功能包安装及使用说明

### 1. 下载
在小车上打开终端，运行：

```
$ git clone https://gitee.com/liuyvjin/speedstar-final.git
$ cd ./speedstar-final/Ros
```

### 2. 安装
运行：
```
$ bash ./setup.sh
```
该脚本将会创建ros环境 `~/speedstar_ws`， 并自动编译功能包。

### 3. 使用
`auto_driver`中的`driver_utils.Driver`为小车驱动类，包含以下功能：
```
    小车驱动类,包含以下功能：
    驱动：
    1. set_direction: [0,100]
    2. set_speed: [0,100]
    3. set_mode: [1D, 2N, 3P, 4R]
    4. set_beep: [0,1]
    读取传感器：
    1. get_acc: imu加速度
    2. get_alpha: imu角加速度
    3. get_B:  imu磁场
    4. get_theta: imu角度
    5. get_speed: 实际电机速度
    6. get_mode: 实际档位
    7. get_direction: 实际方向
    8. get_supersonic: 实际超声波距离
    9. get_sensors: 所有传感器信息
    10. get_lane: 小车偏离路中心距离
    11. get_objs: 目标检测结果
```

具体使用可参考`driver_node.py`：
- 运行基本服务：`roslaunch bluedge_bridge base.launch`
- 运行小车驱动节点：`rosrun bluedge_bridge driver_node.py`

### 4. 终端快捷命令
`ROS/cmd.sh`中定义了一些常用命令，在 `base.launch` 运行时可设置小车方向、速度、档位。
```
.  cmd.sh      # 注册命令
set-speed  0   # 设置速度
set-direct 10  # 设置方向
set-mode   3   # 设置档位 [1D, 2N, 3P, 4R]
```

