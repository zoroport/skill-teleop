"""Terminal controls, retained alongside the Unity Quest service client."""
import argparse
import select
import sys
import termios
import tty

import rclpy
from std_srvs.srv import SetBool, Trigger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", default="left", choices=("left", "right"))
    args = parser.parse_args()
    if not sys.stdin.isatty():
        parser.error("An interactive terminal is required; otherwise call ROS services directly")
    rclpy.init()
    node = rclpy.create_node("skill_operator_keyboard")
    commands = {"r": "/collection/start", "s": "/collection/success", "f": "/collection/failed",
                "g": "/xhand/skill/grasp", "x": "/xhand/skill/start", "h": "/xhand/skill/stop",
                "o": "/xhand/skill/release", "p": "/xhand/skill/capture_release",
                "j": f"/{args.side}/home", "k": f"/{args.side}/capture_home",
                "1": "/collection/mark_approach", "2": "/collection/mark_grasp",
                "3": "/collection/mark_screw", "4": "/collection/mark_release"}
    clients = {k: node.create_client(Trigger, v) for k, v in commands.items()}
    stop_arm = node.create_client(SetBool, f"/{args.side}/stop_teleop")

    def call(client, request):
        if not client.wait_for_service(timeout_sec=1):
            print(f"Unavailable: {client.srv_name}")
            return
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=3)
        if future.done():
            response = future.result()
            print(f"{client.srv_name}: {response.success}, {response.message}")
        else:
            print(f"Timeout: {client.srv_name}; result is unknown, do not assume cancellation")

    print("r录制 s成功结束 f失败结束 | g抓握 x旋拧 h保持 o松开 p保存当前松开姿态")
    print("1接近 2抓握 3旋拧 4松开：阶段标记 | q停止手技能/手臂并结束录制、退出")
    print("j机械臂回home | k将当前静止关节姿态保存为home（覆盖原home）")
    previous = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if not select.select([sys.stdin], [], [], .1)[0]:
                continue
            key = sys.stdin.read(1).lower()
            if key == "q":
                call(clients["h"], Trigger.Request())
                request = SetBool.Request()
                request.data = False
                call(stop_arm, request)
                call(clients["f"], Trigger.Request())
                break
            if key in clients:
                call(clients[key], Trigger.Request())
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, previous)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
