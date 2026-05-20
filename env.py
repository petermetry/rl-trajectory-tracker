import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np

class TrajectoryTrackingEnv(gym.Env):
    def __init__(self, render_mode=None, noise_std=0.005):
        super().__init__()
        self.render_mode = render_mode
        self.noise_std = noise_std
        
        # Setup PyBullet
        self.physics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Action space: 7 DoF joint velocities
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Observation space: 
        # 7 (q) + 7 (dq) + 3 (ee_pos) + 3 (target) + 9 (3 future targets) = 29
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(29,), dtype=np.float32)
        
        self.robot = None
        self.num_joints = 7
        self.ee_link_idx = 11 # Franka hand
        self.t = 0.0
        self.dt = 0.05
        self.prev_action = np.zeros(7)
        
        self._load_world()

    def _load_world(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane = p.loadURDF("plane.urdf")
        # Load Franka Panda
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        
        # Enable velocity control
        for j in range(self.num_joints):
            p.changeDynamics(self.robot, j, linearDamping=0, angularDamping=0)

    def _get_trajectory_point(self, t):
        # 3D Figure-Eight (Lissajous) in front of the robot
        x = 0.5 + 0.15 * np.sin(2 * t)
        y = 0.2 * np.cos(t)
        z = 0.4 + 0.15 * np.sin(t)
        return np.array([x, y, z])

    def _get_obs(self):
        # Get joint states
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        q = np.array([state[0] for state in joint_states])
        dq = np.array([state[1] for state in joint_states])
        
        # Get End-Effector state
        ee_state = p.getLinkState(self.robot, self.ee_link_idx)
        ee_pos = np.array(ee_state[0])
        
        # Add observation uncertainty (Noise)
        ee_pos += np.random.normal(0, self.noise_std, size=3)
        
        # Get current and future targets (lookahead)
        targets = []
        for i in range(4): # t, t+1dt, t+2dt, t+3dt
            targets.append(self._get_trajectory_point(self.t + i * self.dt))
        
        obs = np.concatenate([q, dq, ee_pos] + targets)
        return obs.astype(np.float32)

    def step(self, action):
        # Scale action to actual joint velocities
        scaled_action = action * 2.0 
        
        # Apply velocities
        p.setJointMotorControlArray(
            self.robot,
            range(self.num_joints),
            p.VELOCITY_CONTROL,
            targetVelocities=scaled_action
        )
        p.stepSimulation()
        self.t += self.dt
        
        obs = self._get_obs()
        ee_pos = obs[14:17]
        target_pos = obs[17:20]
        
        # Reward Calculation
        dist = np.linalg.norm(ee_pos - target_pos)
        r_dist = np.exp(-10.0 * dist) # Bound distance reward
        r_smooth = -0.1 * np.linalg.norm(action - self.prev_action)
        r_energy = -0.01 * np.linalg.norm(action)
        
        reward = r_dist + r_smooth + r_energy
        self.prev_action = action.copy()
        
        # Terminate if arm goes out of bounds (instability)
        terminated = bool(dist > 0.5) 
        truncated = bool(self.t > 10.0) # 200 steps
        
        return obs, reward, terminated, truncated, {"dist": dist}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0.0
        self.prev_action = np.zeros(7)
        self._load_world()
        
        # Reset arm to home position
        home_q = [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]
        for j in range(self.num_joints):
            p.resetJointState(self.robot, j, home_q[j])
            
        return self._get_obs(), {}

    def close(self):
        p.disconnect(self.physics_client)