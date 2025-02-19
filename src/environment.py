import gymnasium as gym

def create_environment(env_name="CartPole-v1"):
    """Gym ortamı oluştur ve sıfırla"""
    env = gym.make(env_name)
    return env