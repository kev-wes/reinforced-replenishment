import os
import shutil
from collections import deque
from datetime import datetime

import plotly.graph_objects as go
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.notebook import tqdm


class ProgressBarLoggingCallback(BaseCallback):
    """
    Custom callback to log the average reward of the last 100 episodes during training
    and display it in the progress bar every 100 episodes.
    """

    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = None
        self.episode_rewards = deque(maxlen=100)  # Store the last 100 episode rewards
        self.episode_count = 0  # Track the number of episodes

    def _on_training_start(self) -> None:
        # Initialize the progress bar
        self.progress_bar = tqdm(
            total=self.total_timesteps, desc="Training Progress", leave=True
        )
        self.progress_bar.set_description(f"Training Progress (Avg Reward: {None})")

    def _on_step(self) -> bool:
        # Update the progress bar
        self.progress_bar.update(1)

        # Check if the episode is done
        if "episode" in self.locals["infos"][0]:
            episode_reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(episode_reward)  # Add the reward to the deque
            self.episode_count += 1  # Increment the episode count

            # Update the progress bar description every 100 episodes
            if self.episode_count % 100 == 0:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                self.progress_bar.set_description(
                    f"Training Progress (Avg Reward: {avg_reward:.2f})"
                )
        return True

    def _on_training_end(self) -> None:
        # Close the progress bar
        if self.progress_bar is not None:
            self.progress_bar.close()


class PlotLoggingCallback(BaseCallback):
    """
    Custom callback to log the average reward of the last 100 episodes during training,
    save the model with a timestamp, save an interactive Plotly plot of the rewards,
    and store the current replenishment_env.py code.
    """

    def __init__(
        self,
        total_timesteps,
        model_save_dir="../data/06_models",
        env_file_path="../src/reinforced_replenishment/envs/replenishment_env.py",
        verbose=0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.model_save_dir = model_save_dir
        self.env_file_path = env_file_path  # Path to the environment file
        self.progress_bar = None
        self.episode_rewards = deque(maxlen=100)  # Store the last 100 episode rewards
        self.episode_count = 0  # Track the number of episodes
        self.avg_rewards = []  # Store the average rewards for plotting
        self.episodes = []  # Store the episode numbers for plotting

    def _on_training_start(self) -> None:
        # Initialize the progress bar
        self.progress_bar = tqdm(
            total=self.total_timesteps, desc="Training Progress", leave=True
        )
        self.progress_bar.set_description(f"Training Progress (Avg Reward: {None})")

    def _on_step(self) -> bool:
        # Update the progress bar
        self.progress_bar.update(1)

        # Check if the episode is done
        if "episode" in self.locals["infos"][0]:
            episode_reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(episode_reward)  # Add the reward to the deque
            self.episode_count += 1  # Increment the episode count

            # Update the progress bar every 100 episodes
            if self.episode_count % 100 == 0:
                avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                self.progress_bar.set_description(
                    f"Training Progress (Avg Reward: {avg_reward:.2f})"
                )

                # Store data for plotting
                self.avg_rewards.append(avg_reward)
                self.episodes.append(self.episode_count)
        return True

    def _on_training_end(self) -> None:
        # Close the progress bar
        if self.progress_bar is not None:
            self.progress_bar.close()

        # Generate a timestamp for versioning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a directory for this version
        version_dir = os.path.join(self.model_save_dir, f"version_{timestamp}")
        os.makedirs(version_dir, exist_ok=True)

        # Save the model with the timestamp
        model_path = os.path.join(version_dir, "ppo_replenishment.zip")
        self.model.save(model_path)

        # Save the environment file
        if os.path.exists(self.env_file_path):
            shutil.copy(
                self.env_file_path, os.path.join(version_dir, "replenishment_env.py")
            )

        # Generate the interactive Plotly plot
        fig = go.Figure()

        # Add the average reward line
        fig.add_trace(
            go.Scatter(
                x=self.episodes,
                y=self.avg_rewards,
                mode="lines",
                name="Avg Reward (Last 100 Episodes)",
                line=dict(color="blue"),
            )
        )

        # Customize the layout
        fig.update_layout(
            title="Average Reward Over Episodes",
            xaxis_title="Episodes",
            yaxis_title="Average Reward",
            template="plotly_white",
            legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top"),
        )

        # Save the plot as an HTML file
        plot_path = os.path.join(version_dir, "reward_plot.html")
        fig.write_html(plot_path)
        print(f"Model, reward plot and environment saved to: {version_dir}")
        fig.show()
