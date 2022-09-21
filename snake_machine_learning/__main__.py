from random import randint

import numpy as np
import pygame
from keras.utils import to_categorical

from snake_machine_learning.game.game import display, get_record, initialize_game
from snake_machine_learning.game.game_classes import Game
from snake_machine_learning.game.game_config import game_config
from snake_machine_learning.ml.DQN import DQNAgent

from file_dump import FileDump

def main():
    """Main method to start the game and the learning process"""

    # Initialize Pygame and create a new agent.
    pygame.init()
    # agent = DQNAgent()
    agent = FileDump().load_network()
    if agent is None:
        agent = DQNAgent()

    counter_games = 0
    record = 0

    while counter_games < 1500:
        # Initialize classes
        game = Game()
        snake1 = game.snake
        apple1 = game.apple

        # Perform first move
        initialize_game(snake1, game, apple1, agent)
        # todo, display(snake1, apple1, game, record)
        # display(snake1, apple1, game, record)

        while not game.crash:
            # Agent.epsilon is set to give randomness to actions
            # agent.epsilon = 80 - counter_games
            agent.epsilon = (150 - counter_games)*200/150/30

            # Get old state
            state_old = agent.get_state(game, snake1, apple1)

            # Perform random actions based on agent.epsilon, or choose the action
            if randint(0, 200) < agent.epsilon:
                final_move = to_categorical(randint(0, 2), num_classes=3)
            else:
                # Predict action based on the old state
                prediction = agent.model.predict(state_old.reshape((1, 11)))
                final_move = to_categorical(np.argmax(prediction[0]), num_classes=3)

            # Perform new move and get new state
            snake1.do_move(final_move, snake1.x, snake1.y, game, apple1)
            state_new = agent.get_state(game, snake1, apple1)

            # Set reward for the new state
            reward = agent.set_reward(snake1, game.crash)

            # Train short memory base on the new action and state
            agent.train_short_memory(state_old, final_move, reward, state_new, game.crash)

            # Store the new data into a long term memory
            agent.remember(state_old, final_move, reward, state_new, game.crash)
            record = get_record(game.score, record)
            # todo, display(snake1, apple1, game, record)
            # display(snake1, apple1, game, record)
            pygame.time.wait(game_config["speed"])

        agent.replay_new(agent.memory)
        counter_games += 1
        print("Count", counter_games, "      Score:", game.score, "      Record:", record)
        if counter_games % 10 == 0:
            FileDump().save_network(agent)
            print("Network saved.")


if __name__ == "__main__":
    main()

'''
ml_config.py 에서  "rewards": {"negative": -10, "positive": 10}, negative -10, positive 10 으로 하면 생존에만 신경 쓰는 것 같다.
이 상태에서 학습 회수를 1000, 10000으로 늘려서 계속 학습만 하면 high score 50까지 찍고 그 후는 생존하려고 뱅뱅이만 돈다. 그래서 score가 1, 2 로만 나온다.

일단 한번 실행하고 끝내면 처음부터 다시 실행 해야 한다. network이 초기화 되기 때문이다.
그래서 file_dump.py를 만들어 state를 저장하게 했다. network.bin 이 100MB 가 넘어가고 있다. 왜 학습할 수록 더 커질까? 커질 이유가 없지 않나? 
garbage collection 되어야 할 것도 함께 저장되는 것 아닐까?   

positive를 20으로 늘리니 하니 먹이를 찾는 것 같다. 그러나 최대 score가 50 정도이다.

OpenAI Gym 을 사용 해 보자.
reinforcement learning q-learning
 https://www.youtube.com/watch?v=Fbf9YUyDFww&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=15
 https://mclearninglab.tistory.com/49 링크 모음
- OpenAI Gym
    https://github.com/openai/gym
- OpenAI Gym snake
  https://github.com/grantsrb/Gym-Snake
  https://pypi.org/project/snake-gym/
  https://github.com/vivek3141/snake-ai
  https://github.com/yosinlpet/gym_snake

'''