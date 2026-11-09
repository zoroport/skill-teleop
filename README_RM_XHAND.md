# RM＋Xhand＋D435：Unity 遥操作示教与 BC/DP 训练

**训练后的四策略部署**已新增独立入口，见 [部署与验证指南](../minbc/DEPLOY_RM_XHAND.md)。
下文记录的是采集和训练流程；部署时退出遥操作、手部技能节点和采集器，使用独立硬件桥。

本接入使用左侧 RM、单 Xhand、外部固定 D435。Unity → ROS-TCP Endpoint →
原 RM 遥操作节点 → RM SDK 的控制链路保留；独立采集器订阅实际反馈与命令。
手部使用已验证的 `nutbolt_1019.pt`，以 20 Hz 执行 RL 旋拧技能。
采集、数据转换和训练采用新入口，不需要运行原 UR/ZMQ 的 `launch_nodes.py`、
`run_env.py` 或 `oculus_reader`。

代码已完成离线联调；具体检查及未进行的真机验收见 [验证记录](VALIDATION_RM_XHAND.md)。

## 1. 接入内容与边界

```text
Unity / Quest → ROS-TCP Endpoint → RM 遥操作 → RM
                                    ↓ 状态、原始输入、下发目标/返回码
键盘/服务 → Xhand 技能节点 → Xhand    ↓
                 ↓ 状态、命令     独立采集器 ← D435 RGB/对齐深度
                 └───────────────→ 原始 episode
                                    ↓ 离线对齐、质检、转换
                                minbc BC / DP
```

- 原 Unity 话题 `/left/vr_arm_delta_pose` 和 `/left/start_teleop`、`/left/stop_teleop` 服务不变。
- 新手臂参数 `enabled_sides` 默认 `[left,right]`；采集配置改为 `[left]`，不会连接右臂。
- `telemetry_enabled` 默认 false。关闭时，原手臂运行不依赖新消息包。
- RM 实时反馈使用 SDK 自带的 UDP 上报，再由节点发布为 ROS2 消息。这只用于反馈，
  不改变 Unity 的 ROS-TCP 控制方式。SDK 三线程模式为 `thread_mode=2`。
- 所有采集进程必须位于同一 Linux 主机。主机名和 boot ID 不一致时拒绝录制。
- 本轮实现采集和离线训练；生成的 21 维动作不能直接交给原 UR 关节部署入口。

## 2. 一次性环境准备

仓库保持目前的并列结构：

```text
/home/cc/Wlh_Projects/01_dexscrew/
  dexscrew/xhand-deploy/{nut_policy.py,checkpoints/nutbolt_1019.pt}
  skill-teleop/
  minbc/
/home/cc/Wlh_Projects/02_HandAndArm_control/Arm/
```

### ROS 消息包

使用 ROS2 Humble 的系统 Python 构建：

```bash
source /opt/ros/humble/setup.bash
cd /home/cc/Wlh_Projects/01_dexscrew/skill-teleop/ros2_ws
colcon build --packages-select skill_teleop_interfaces \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

需要系统已有 `colcon`、`ament_cmake`、`rosidl_default_generators`、`rclpy`、
`geometry_msgs`、`std_msgs` 和 `std_srvs`。本机已经完成该消息包的编译及序列化测试。

### 手部、相机和采集环境

保留已验证的 `xhand_117` 环境，在其上创建独立 venv，继承原 Torch/SDK，
只向新环境增加采集依赖：

```bash
cd /home/cc/Wlh_Projects/01_dexscrew/skill-teleop
/home/cc/miniforge3/envs/xhand_117/bin/python -m venv --copies --system-site-packages .venv-collection
source .venv-collection/bin/activate
python -m pip install -r requirements-collection.txt
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python -m skill_teleop.preflight --component hand
python -m skill_teleop.preflight --component collector
```

`--copies` 很重要：现有 `xhand_117` 解释器带 `cap_net_raw`，直接启动它可能导致
动态加载器忽略 ROS 的库路径。新解释器副本不继承该文件能力；本次 RS485 不需要它。
不要给新的采集解释器添加该能力。预检查仅导入依赖和检查文件，不打开相机或机器人。

每个新终端都需要激活对应环境，并 source ROS 和消息包。不要用新的 `PYTHONPATH=...`
覆盖 ROS setup 已设置的路径。所有真机进程与现有 ROS-TCP Endpoint 使用同一 ROS domain。

## 3. 真机启动顺序

以下命令属于现场操作。代码开发和离线测试没有执行这些真机启动命令。

### 终端 A：保留现有 Unity / ROS-TCP Endpoint

按目前已成功复现的方式启动 Unity 与 ROS-TCP Endpoint。仅用键盘控制技能时可沿用原 Unity 客户端。
如要使用 Quest 技能/home 按键，请替换本轮更新的 `unity_code/LeftHandle.cs`，
按 [Quest 接入说明](../unity_code/README.md) 配置；键盘方式依然可用。
若左臂遥操作正常，但右手技能键/左 X 超时且 Endpoint 报 `Trigger_Request` Fast CDR
反序列化错误，请同步新版 `LeftHandle.cs`（内置 ROS2 空请求兼容处理），并按上述说明重启连接。
不需要修改从端服务或重新 colcon 构建。

### 终端 B：仅左侧 RM，并开启遥测

在现有、已经能运行机械臂的 `arm_hand118_py310` 环境中：

```bash
conda activate arm_hand118_py310
source /opt/ros/humble/setup.bash
source /home/cc/Wlh_Projects/01_dexscrew/skill-teleop/ros2_ws/install/setup.bash
python /home/cc/Wlh_Projects/02_HandAndArm_control/Arm/dual_arm_teleop_node.py \
  --ros-args --params-file /home/cc/Wlh_Projects/01_dexscrew/skill-teleop/configs/rm_left_collection.yaml
```

参数文件默认 RM `192.168.1.19:8080`、反馈 UDP 端口 8090，主机 IP 根据到 RM 的路由选择。
多网卡时在配置中明确设置 `telemetry_host_ip`。节点会读取关节数、工作系和工具系；
不要在同一采集过程中改动这些坐标系配置。正常关闭时恢复之前的 UDP 上报配置。
RM 重连后遥测标为失效，需要重新启动手臂采集节点以复核坐标系；原遥操作的重连行为保留。
原始元数据还保存 `rm_movep_canfd` 的 follow、trajectory_mode、radio 及遥操作限幅参数。

### 终端 C：Xhand 技能

```bash
cd /home/cc/Wlh_Projects/01_dexscrew/skill-teleop
source .venv-collection/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python -m skill_teleop.hand_node --hardware \
  --serial-port /dev/ttyUSB0 --baud-rate 3000000 --hand-id 22 --hz 20
```

只有加 `--hardware` 才访问串口；不加时是合成手部反馈。启动会读取当前位置并发送保持目标，
不会自动抓握或旋拧。新节点是串口的唯一控制者：不要同时运行旧部署脚本或
`02_HandAndArm_control/Hand/hand_teleop.py`。同一新节点启动两份时，串口锁会拒绝第二份。

抓握用现有螺母初始目标，插值约 1 秒；实际反馈到达目标误差小于 0.1 rad 后才允许旋拧。
每轮读取一次整手反馈，策略、状态发布和触觉读取共享该反馈。异常进入 FAULT，停止继续下发，
处理故障后重启节点。保持和关闭都不会自动松手或关闭电机。

触觉默认尝试读取并随原始数据保存，无需启用开关。SDK 未返回可用触觉时标记
`tactile_valid=false`，不填零冒充测量。默认 BC/DP 不使用触觉；要训练触觉模态，
转换和训练两步均需添加 `--tactile`。

### 松开姿态配置

默认松开目标已设为螺母旋拧策略的初始关节目标 `NutPolicy.initial_hardware`，
与 `g` 的最终抓握目标相同。按 `o` 会约 1 秒插值返回此姿态，随后保持；
这个定义表示回到初始姿态，不保证物体实际脱离手指。
无需先配置文件即可使用。SDK 硬件顺序下的目标（rad）为：

```text
[1.3, 0.6, 0.45, -0.17, 1.1, 0.4, 1.1, 0.4, 0, 0, 0, 0]
```

如需覆盖默认目标，可用已经验证的手部控制方法设置合适的姿态，停止该控制程序，再启动新技能节点。
节点此时保持实际姿态，按键 `p` 将其保存到 `calibration/xhand22_release.json`。
也可以准备以下格式的文件（角度必须来自现场确认的姿态）：

```text
{"hand_id":22,"order":"hardware","positions":[12 个 SDK 硬件顺序的关节角，单位 rad]}
```

可用 `--release-file /绝对路径/pose.json` 指定文件。节点检查手 ID、顺序、有限值与机械限位。

### 终端 D：D435 与独立录制

```bash
cd /home/cc/Wlh_Projects/01_dexscrew/skill-teleop
source .venv-collection/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python -m skill_teleop.collect --side left --data-dir data/rm_xhand_raw
```

只有一台相机时自动选择；多台时添加 `--camera-serial 实际序列号`。
相机为 640×480、30 FPS，预热约 1 秒，保存 RGB 和对齐到 RGB 的原始 uint16 深度。
深度米数 = 原始深度 × 元数据 `depth_scale`。不做滤波和灰度化。

### 终端 E：操作工具

```bash
cd /home/cc/Wlh_Projects/01_dexscrew/skill-teleop
source .venv-collection/bin/activate
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
python -m skill_teleop.keyboard --side left
```

| 按键 | 行为 | 对应 Trigger 服务 |
|---|---|---|
| r | 开始新 episode | `/collection/start` |
| s / f | 成功 / 失败结束本条录制 | `/collection/success` / `/collection/failed` |
| g | 抓握到螺母初始姿态 | `/xhand/skill/grasp` |
| x / h | 开始旋拧 / 停止并保持 | `/xhand/skill/start` / `/xhand/skill/stop` |
| o / p | 松开 / 保存当前松开姿态 | `/xhand/skill/release` / `/xhand/skill/capture_release` |
| j / k | 机械臂回 home / 将当前静止关节姿态保存为 home | `/left/home` / `/left/capture_home` |
| 1 / 2 / 3 / 4 | 标记接近 / 抓握 / 旋拧 / 松开阶段 | `/collection/mark_approach` 等 |
| q | 停手技能、调用原 RM 停止服务、失败结束录制并退出工具 | 依次调用上述服务 |

也可直接调用，例如 `ros2 service call /xhand/skill/start std_srvs/srv/Trigger '{}'`。
`s/f` 只结束录制，不停止设备；`h` 只停止旋拧，不结束录制。退出采集器不会干涉控制节点。
阶段标记 `1/2/3/4` 完全可选。不按这些键不影响时间对齐、数据有效性检查、转换或 BC/DP 训练；
质检报告只是列出已有事件，不要求四个阶段齐全。未手动标注的阶段不应作为人工阶段真值使用。

## 4. 一条完整示教

1. 确认相机视野覆盖整个过程、技能节点处于 HOLD，并确认默认松开/返回初始姿态符合现场需要。
   HOLD 指 Xhand 暂停运行旋拧策略、按 20 Hz 持续下发最后的关节目标；不是断电或停止发布。
   节点启动时保持启动时读到的姿态；抓握/松开到位后或按右 A（键盘 `h`）也会进入 HOLD。
   HOLD 本身不表示已完成抓握，开始旋拧前仍需执行抓握并等待完成。
2. 用现有 Unity 启动左臂遥操作，并先产生至少一个成功的位姿命令。
   当前按键方案需要持续按住左 trigger；收到启动成功后，手柄不动也会发送目标，无需刻意移动机械臂。
   录制必须同时收到未过期的 RM/手/相机反馈以及当前会话有效的动作目标，缺任一来源会拒绝 `r`。
   “未过期”检查时间戳，不要求反馈数值变化；机械臂静止、只手部旋拧可以正常录制。
3. 按 `r` 开始录制，接近目标后按 `g`；抓握完成后按 `x`。
4. 旋拧结束按 `h`，之后按 `o` 返回配置的松开姿态；完整过程成功后按 `s`，失败按 `f`。
   如需要阶段分析，可额外按 `1/2/3/4` 标记，不影响上述操作。
5. 手臂静止时保持遥操作会话有效，最后接受的目标会作为保持动作。
   当前版本松开左 trigger 会调用手臂 stop，不能把“松开扳机”当作可录制的静止保持模式。
   持续按住时机械臂仍会跟随手柄，不是锁定位置；需要停止时应正常停止，不能为保留标签而忽略停止需求。
   若调用原手臂 stop 服务，由于慢停后的实际停点不能当作原目标，转换会切断该片段，
   直到重新 start 并产生有效命令。不会伪造一个“已下发”的停止姿态。

训练前建议先采两条短示教检查数据，再扩大采集量。两条只是能形成独立 train/test 的最低数量，
不是任务学习所需的数据量承诺。设备动作到达、相机覆盖和任务成功都需要现场验收。

## 5. 文件、质检和转换

### 保存松开姿态时报关节越界

`/xhand/skill/capture_release` 检查最新一次整手反馈 `self.measured`，按 SDK
`finger_state[0:12]` 顺序、弧度单位校验，要求当前模式为 HOLD。当前固定范围如下：

| SDK 下标（从 0 开始） | 最小值 rad | 最大值 rad |
|---|---:|---:|
| 0 | 0 | 1.83 |
| 1 | -1.05 | 1.57 |
| 2 | -0.17 | 1.83 |
| 3 | -0.175 | 0.175 |
| 4–11（每个关节） | 0 | 1.92 |

这些是 `nut_policy.py` 中的配置范围，经策略顺序转换为 SDK 顺序；不是运行时查询到的
真实硬件物理限位，也不完全等于 `02_HandAndArm_control/Hand/urdf` 的限位。
保存松开姿态时允许最多 **3°（约 0.05235988 rad，包含边界）**的反馈越界：
低于下限不超过 3°时保存为下限，高于上限不超过 3°时保存为上限；范围内的值原样保存。
任一关节超出超过 3°，整次保存被拒绝，原松开姿态保持不变。
例如关节 9 为 -2.1666°、关节 11 为 -0.166662°时，两者都保存为 0。
容差仅用于 `capture_release`，不扩大控制目标限位，不修改原始反馈或采集记录。
抓握、松开目标及加载已保存姿态仍使用严格限位检查。

新版错误会列出所有越界关节下标、实测值（rad/deg）、允许范围和超出量。
更新 `hand_node.py` 后重启手部技能节点即可，无需重新编译 ROS 接口。
保存成功的响应会列出被裁剪的关节及裁剪前后值。标定文件中的 `positions` 是合法目标，
同时保存 `measured_positions`、`clipped_joint_indices` 和 `capture_tolerance_deg` 以便追溯。
也可读取整手反馈辅助核对（这里是实测位置，不是下发目标）：

```bash
ros2 topic echo /xhand/collection/state --once --field positions
```

```text
data/rm_xhand_raw/<日期_时间_唯一ID>/
  metadata.json     # 配置、时间基准、状态；中断时不会变成 success
  quality.json      # 录制计数、无效记录、各源时间间隔统计与写盘错误
  chunks/000000.pkl # 各源事件的列表，含 stream、时间戳、序号、有效性
```

写盘使用后台线程、有限队列和临时文件原子替换。溢出/写盘错误使 episode 成为 incomplete。
异常断电留下 recording 状态及已完成的数据块，转换器默认排除，不自动宣称成功。
RGB 原始采集约 28 MB/s，深度约 18 MB/s，另有少量状态数据；一分钟约 2.8 GB，
请先按采集时长安排空间。转换器按块懒加载相机，避免同时载入整条原分辨率视频。

```bash
python -m skill_teleop.quality data/rm_xhand_raw/实际episode目录
python -m skill_teleop.convert data/rm_xhand_raw data/rm_xhand_dataset
# 需要将触觉用于训练时，转换时就强制检查完整性：
python -m skill_teleop.convert data/rm_xhand_raw data/rm_xhand_tactile --tactile
```

输出目录必须不存在，避免混合旧数据。工具仅处理本机可信采集文件，pickle 不用于交换不可信数据。

转换按 20 Hz 网格执行，默认最大观测年龄 100 ms：

- 观测取 `t` 之前的最新样本；动作取 `[t,t+50ms)` 最后成功下发目标，无新目标则保持。
- 无效状态、命令失败、丢序号、会话变化、时间断层切断序列；不会跨断层插值。
- D435 彩色/深度设备帧号跳变也会标为无效，不能仅凭采集器本地序号连续判断无丢帧。
- 不按手臂运动裁剪，静止旋拧与等待保留。DP 所需的最短连续段为 8 帧，更短段计入报告。
- 按原 episode 划分 90% train / 10% test，seed=0；同一 episode 的所有片段归于同一组。
- success 以外的原始示教保留但不进入默认训练集。两组必须均有可用连续片段，否则转换失败。
- 坐标系、关节顺序和相机标定不一致时拒绝混合。`dataset.json` 保存动作定义、布局、划分及丢帧原因。

动作 21 维：`[位置xyz(3), 旋转矩阵第1列(3), 第2列(3), Xhand硬件顺序关节目标(12)]`。
位置在 RM 当前工作坐标系、单位米；手关节单位弧度。原始末端状态以 xyzw 四元数保存。
`skill_teleop.schema.action_to_pose` 对网络旋转输出作正交化，并拒绝退化向量。
SDK 返回成功只表示命令下发成功，不等同于物理到达。

### 排查：未启动 Unity 与录制拒绝

RM 的 `arm_state` 来自 SDK 实时上报，启动终端 B 后即可发布，不依赖 Unity。
`{'err_len': 1, 'err': ['0']}` 是含一个零代码的反馈，不能把条目数量 1 当成故障。
遥测现在逐项检查代码：零代码接受，非零代码及格式异常仍将状态标为无效。

即使反馈正常，未启动遥操作并成功下发目标时也没有有效 `arm_command`。
录制要求反馈和动作标签都有效，因此 `r` 会被拒绝；随后按 `s` 会返回
`No episode is recording`，表示此前尚未开始录制，而不是结束时丢失了文件。
当前没有绕过动作标签检查的录制模式。可只读检查反馈：

```bash
ros2 topic echo /left/collection/arm_state --once --field valid
```

若提示 `Waiting for valid arm_command: left teleoperation stopped; arm slow-stop command sent`，
表示最近已停止左臂遥操作（例如松开左 trigger、追踪丢失或调用 stop 服务），并成功下发慢停。
采集器定时重复报告该状态，并未重复发送慢停。慢停终点可能不同于此前的位姿目标，因此旧目标已失效。
重新松开再按住左 trigger，等待启动成功及首条成功目标后，再按 `r`。
不要求机械臂位置变化；但若再次松开扳机，开始录制仍会被拒绝。
若录制中停止遥操作，原始数据继续保存，但停止期间没有有效手臂动作标签，离线转换会排除并切断该段。

手部 HOLD 状态可只读确认：

```bash
ros2 topic echo /xhand/collection/state --once --field mode
```

HOLD 是完整示教前的建议起始状态，并非采集器只允许此模式；GRASP、SCREW、RELEASE
期间只要反馈、动作及其他数据源有效，也能录制。右 A 仅将手部切换为 HOLD，不会停止左臂遥操作或结束录制。

Ctrl+C/SIGTERM 退出机械臂进程时，现在先清理 SDK/遥测并恢复 UDP 上报配置，
再关闭 ROS context，避免清理日志向已失效的 rosout publisher 发布。
更新这两处 Python 代码后重启终端 B 即可，不需要重新编译消息包。

## 6. BC 与 DP 训练

训练环境与 ROS 控制环境分开。一个已离线验证的 CPU 组合为 Python 3.10、
Torch 2.4.0+cpu、torchvision 0.19.0+cpu、diffusers 0.30.3。
GPU 训练使用配套的 CUDA Torch/torchvision；训练脚本默认 GPU 0，不会自动安装或升级 CUDA。

```bash
cd /home/cc/Wlh_Projects/01_dexscrew/minbc
# 在独立训练环境中，先安装配套 torch/torchvision，再安装：
python -m pip install -r requirements-rm.txt
python train_rm.py --dataset ../skill-teleop/data/rm_xhand_dataset --policy bc --name rm_bc
python train_rm.py --dataset ../skill-teleop/data/rm_xhand_dataset --policy dp --name rm_dp
```

RM 训练入口默认启用训练随机裁剪和验证居中裁剪（`data.scratch_crop=true`），将 320×240 RGB 裁至 288×216，宽高各保留 90%。带触觉/无触觉、BC/DP 均适用。原始采集和数据转换仍保留全幅画面，已有转换数据无需重新导出。

默认 RGB＋RM 关节＋末端位姿＋手关节观测，scratch ResNet，历史 2 帧、预测 16 帧、
动作窗口 8 帧，batch size 32、300 epochs、学习率 0.0002；DP 采用 DDPM、100 个扩散步。
自动从数据清单读取 RM 的 6/7 关节维度。要使用触觉，对经过 `--tactile` 转换的数据加
训练参数 `--tactile`。深度保留在数据中，本轮不加入网络。

先用真实数据运行短程检查：

```bash
python train_rm.py --dataset ../skill-teleop/data/rm_xhand_dataset \
  --policy bc --device cpu --no-images --epochs 1 --batch-size 2 --workers 0 --name cpu_check
```

输出包含 `config.json`、`dataset.json`、`stats.pkl`、`norm.pkl`、EMA checkpoints 和 TensorBoard 日志。
验证集只用训练集统计；损坏或丢失的 step 会报错，不会静默跳过并拼接轨迹。
已有 checkpoint 的输出目录不允许覆盖。CPU 小测试的 loss 和 MSE 不代表真机成功率。

## 7. 开发验证与现场待验收项

已验证：ROS 消息构建/序列化、静止旋拧数据保留、时间对齐、6/7 关节、单相机、触觉可选、
写盘失败/溢出/中断、原始策略连续 40 步等价、真实 ROS 传输下的模拟节点服务和录制、
BC/DP CPU 视觉训练与 checkpoint 重载输出 `(8,21)`。DP 快速测试使用 4 个扩散步，正式预设仍为 100。

复现离线验证（分别运行，避免假 SDK 导入与真实 ctypes 测试互相影响）：

```bash
cd /home/cc/Wlh_Projects/01_dexscrew/skill-teleop
python -m unittest discover -s tests -p test_collection.py -v
python -m unittest discover -s tests -p test_arm_session.py -v
python -m unittest discover -s tests -p test_arm_home.py -v
python -m unittest discover -s tests -p test_nut_policy.py -v
# 仅本机独立 ROS 域，RM/相机/手全部为模拟后端：
ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=217 python -m unittest discover -s tests -p test_ros_integration.py -v

cd /home/cc/Wlh_Projects/01_dexscrew/minbc
python -m unittest discover -s tests -p test_rm_dataset.py -v
python tests/smoke_rm_training.py --output /tmp/rm_training_smoke_new
```

ROS 集成测试额外需要 RM SDK 的 ctypes 类型；测试脚本在本机可读取 `arm_hand118_py310` 的安装。
它替换了全部 SDK 设备调用，不连接机械臂。训练测试创建合成数据，输出目录必须是新的。

现场仍需确认：实际 RM 型号与坐标系、D435 帧率/视野、Xhand 触觉可用性、松开姿态，
以及开启采集前后的遥操作延迟和手部 20 Hz 时序。用 `quality` 的间隔统计、SDK 返回码和
真实反馈联合检查，再采完整成功/失败示教。真机训练效果和后续策略部署尚未验收。
