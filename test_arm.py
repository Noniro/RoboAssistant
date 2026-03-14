from lerobot.robots.utils import make_robot_from_config
from lerobot.robots.so101_follower.so101_follower import SO101FollowerConfig

def test_arm():
    print("Testing SO-ARM 101 connection...")
    try:
        cfg = SO101FollowerConfig(port='COM4', id='so_arm_101')
        robot = make_robot_from_config(cfg)
        robot.connect(calibrate=False)
        
        print("Successfully connected!")
        obs = robot.get_observation()
        print(f"Current Observation Keys: {list(obs.keys())[:5]} ...")
        
        robot.disconnect()
        print("Disconnected successfully.")
    except Exception as e:
        print(f"Error connecting to arm: {e}")

if __name__ == "__main__":
    test_arm()
