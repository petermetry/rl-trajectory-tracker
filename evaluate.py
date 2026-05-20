import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from env import TrajectoryTrackingEnv
import time

def evaluate():
    env = TrajectoryTrackingEnv(render_mode="human", noise_std=0.002)
    
    try:
        model = PPO.load("models/best_model")
    except FileNotFoundError:
        print("Pre-trained model not found. Run train.py first.")
        return

    obs, _ = env.reset()
    
    actual_path = []
    target_path = []
    errors = []
    
    print("Running evaluation...")
    for _ in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Extract ground truth without noise for plotting
        ee_pos = env._get_obs()[14:17] # We extract cleanly for logging
        target_pos = obs[17:20]
        
        actual_path.append(ee_pos)
        target_path.append(target_pos)
        errors.append(info["dist"])
        
        time.sleep(0.05) # Slow down for viewing
        
        if terminated or truncated:
            break

    env.close()

    # Calculate metrics
    actual_path = np.array(actual_path)
    target_path = np.array(target_path)
    mean_error = np.mean(errors)
    max_error = np.max(errors)
    
    print(f"Tracking Complete.")
    print(f"Mean Tracking Error: {mean_error:.4f} m")
    print(f"Max Tracking Error: {max_error:.4f} m")

    # Plot 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(target_path[:, 0], target_path[:, 1], target_path[:, 2], 
            label='Target Trajectory', color='blue', linestyle='--')
    ax.plot(actual_path[:, 0], actual_path[:, 1], actual_path[:, 2], 
            label='End-Effector Path', color='red', linewidth=2)
    
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title(f'3D End-Effector Tracking\nMean Error: {mean_error*100:.2f} cm')
    ax.legend()
    
    plt.savefig('tracking_result.png')
    print("Plot saved as tracking_result.png")
    plt.show()

if __name__ == "__main__":
    evaluate()