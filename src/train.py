import os
import torch
from src.environment import create_environment
from src.agent import DQNAgent
from src.utils import ReplayBuffer, TrainingMetrics

def train_model(episodes=500, batch_size=64):
    # Proje kök dizinini bul
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path = os.path.join(project_root, 'models')
    
    env = create_environment("CartPole-v1")
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
    replay_buffer = ReplayBuffer(capacity=50000)
    metrics = TrainingMetrics()
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_length = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            episode_length += 1
            
            loss = agent.train_step(replay_buffer, batch_size)
            if loss is not None:
                metrics.add_loss(loss)
            
        agent.update_epsilon()
        if episode % 10 == 0:
            agent.update_target_network()
        
        metrics.add_episode(total_reward, episode_length)
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Length: {episode_length}")
    
    # Modeli kaydet
    model_path = os.path.join(models_path, 'trained_model.pth')
    torch.save(agent.policy_net, model_path)
    print(f"Model kaydedildi: {model_path}")
    
    return metrics