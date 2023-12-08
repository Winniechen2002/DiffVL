import time
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import wandb

class RewardLengthCallback(BaseCallback):
    def __init__(self):
        super(RewardLengthCallback, self).__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'].sum()
        self.current_episode_length += len(self.locals['dones'])
        
        if any(self.locals['dones']):
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        return True

class WandbCallback(BaseCallback):
    # n_steps = 0

    def __init__(self, check_freq, reward_length_callback = None):
        super(WandbCallback, self).__init__()
        self.check_freq = check_freq
        self.last_time = time.time()
        self.reward_length_callback = reward_length_callback
        self.n_steps = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Collect additional metrics
            self.n_steps += self.check_freq
            mean_episode_reward = np.mean(self.reward_length_callback.episode_rewards[-100:])
            num_episodes = len(self.reward_length_callback.episode_lengths)
            mean_episode_length = np.mean(self.reward_length_callback.episode_lengths[-100:])
            fps = self.check_freq / (time.time() - self.last_time)
            self.last_time = time.time()

            # Log metrics to WandB
            wandb.log({
                'rollout/ep_len_mean': mean_episode_length,
                'rollout/ep_rew_mean': mean_episode_reward,
                'time/episodes': num_episodes,
                'time/fps': fps,
                'time/time_elapsed': self.num_timesteps,
                'time/total_timesteps': self.model.num_timesteps,
                }, step=self.n_steps)
            # If your model is an instance of OffPolicyAlgorithm, you can log training losses as follows

            policy = self.model.policy
            if hasattr(policy, 'actor_loss') and policy.actor_loss is not None:
            # print(policy.actor_loss)
                wandb.log({'train/actor_loss': policy.actor_loss.item()}, step=self.n_steps)
            if hasattr(policy, 'critic_loss') and policy.critic_loss is not None:
                wandb.log({'train/critic_loss': policy.critic_loss.item()}, step=self.n_steps)
            if hasattr(policy, 'ent_coef') and policy.ent_coef is not None:
                wandb.log({'train/ent_coef': policy.ent_coef.item()}, step=self.n_steps)
            if hasattr(policy, 'ent_coef_loss') and policy.ent_coef_loss is not None:
                wandb.log({'train/ent_coef_loss': policy.ent_coef_loss.item()}, step=self.n_steps)

        info = self.locals.get("info")
        if info is not None:
            for key, value in info.items():
                wandb.log({f"info/{key}": value}, step=self.n_calls)
            
        return True


from stable_baselines3.common.callbacks import EvalCallback

class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, sample_steps=100, video_output_path="output.mp4", **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_steps = sample_steps
        self.video_output_path = video_output_path

    def _on_step(self) -> None:
        super()._on_step()
        if self.n_calls % self.eval_freq == 0:
            self.generate_video()

    def generate_video(self):
        vec_env = self.eval_env
        obs = vec_env.reset()
        images = []
        for i in range(self.sample_steps):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, info = vec_env.step(action)
            info = info[0]
            images.append(vec_env.render(mode='rbga'))
            # print(info)
            for key, value in info.items():
                # print(key, value)
                wandb.log({f"reward/{key}": value}, step = self.num_timesteps + i)

        from tools.utils import animate
        animate(images, filename=self.video_output_path)


# eval_callback = EvalCallback(
#     eval_env,
#     eval_freq=1000,  # 根据需要调整评估频率
#     n_eval_episodes=5,
#     deterministic=True,
#     render=False,
#     eval_fn=custom_eval,
# )