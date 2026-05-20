from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from env import TrajectoryTrackingEnv
import os

def main():
    # Setup directories
    models_dir = "models"
    logs_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Initialize Environment
    env = TrajectoryTrackingEnv(render_mode=None, noise_std=0.005)
    check_env(env) # Validate standard Gym API

    # Define PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        tensorboard_log=logs_dir
    )

    # Callback to save best model
    eval_env = TrajectoryTrackingEnv(render_mode=None, noise_std=0.0)
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=models_dir,
        log_path=logs_dir, 
        eval_freq=5000,
        deterministic=True, 
        render=False
    )

    print("Starting training...")
    model.learn(total_timesteps=500_000, callback=eval_callback)
    
    model.save(f"{models_dir}/final_model")
    print("Training complete.")

if __name__ == "__main__":
    main()