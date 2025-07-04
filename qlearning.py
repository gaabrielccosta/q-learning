import numpy as np
import random
import tkinter as tk



# Variáveis de configuração
GUI_SIZE = 50
UPDATE_INTERVAL = 1 # milissegundos
EPISODES = 2000




# Usamos None para posições inválidas
R = [
    [1, 1, 1, 1, -100, 1, 1, 1, 1, 1, 1, -100],
    [1, -100, 1, 1, 1, 1, 1, 1, 1, 1, 1, -100],
    [-100, 1, -100, 1, 1, 1, -100, 1, -100, -100, 1, 1],
    [1, -100, 1, 1, 1, 1, 1, 1, -100, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 100],
    [None, None, None, None, 1, 1, -100, 1, None, None, None, None],
    [None, None, None, None, 1, 1, 1, 1, None, None, None, None],
    [None, None, None, None, 1, 1, 1, 1, None, None, None, None],
    [None, None, None, None, 1, 1, -100, 1, None, None, None, None],
    [None, None, None, None, 1, 1, 1, 1, None, None, None, None],
]

# Dimensões e ações
NUM_ROWS = len(R)
NUM_COLS = max(len(row) for row in R)

ACTIONS = ["N", "S", "L", "O"]
ACTION_DIRECTIONS = {"N": (-1, 0), "S": (1, 0), "L": (0, 1), "O": (0, -1)}

START = (9, 4)
GOAL = (4, 11)


class QLearningAgent:
    def __init__(self, gamma):
        # Inicialização da tabela
        self.Q = {}
        for i, row in enumerate(R):
            for j, cell_value in enumerate(row):
                if cell_value is None:
                    continue

                state = (i, j)
                self.Q[state] = np.zeros(len(ACTIONS), dtype=float)
        self.gamma = gamma

    def choose_action(self, state):
        # 70% dos casos: escolhe a ação de maior valor em Q (exploração do conhecimento)
        if random.random() < 0.7:
            qvals = self.Q[state]
            max_q = np.max(qvals)
            best_actions = [i for i, v in enumerate(qvals) if v == max_q]
            return random.choice(best_actions)
        # 30% dos casos: escolhe ação aleatória (exploração)
        return random.randrange(len(ACTIONS))

    def update(self, s, a, r, s2):
        target = r + self.gamma * np.max(self.Q[s2])
        self.Q[s][a] = target


# Função de movimento
def step(state, action):
    di, dj = ACTION_DIRECTIONS[ACTIONS[action]]
    ni, nj = state[0] + di, state[1] + dj
    if 0 <= ni < NUM_ROWS and 0 <= nj < len(R[ni]) and R[ni][nj] is not None:
        return (ni, nj)
    return state


class VisualApp:
    def __init__(self, agent, episodes):
        self.agent = agent
        self.episodes = episodes
        self.root = tk.Tk()
        self.canvas = tk.Canvas(
            self.root, width=NUM_COLS * GUI_SIZE, height=NUM_ROWS * GUI_SIZE
        )
        self.canvas.pack()
        self.draw_grid()
        self.agent_vis = self.canvas.create_oval(0, 0, GUI_SIZE, GUI_SIZE, fill="red")
        self.info_id = self.canvas.create_text(
            5, 5, anchor="nw", font=("Arial", 10), text=""
        )
        self.reset_episode()
        self.root.after(0, self.train_step)
        self.root.mainloop()

    def draw_grid(self):
        for i, row in enumerate(R):
            for j, val in enumerate(row):
                if val is None:
                    continue
                x, y = j * GUI_SIZE, i * GUI_SIZE
                color = "white"
                if (i, j) == START:
                    color = "blue"
                if (i, j) == GOAL:
                    color = "green"
                if val < 0:
                    color = "gray"
                self.canvas.create_rectangle(
                    x, y, x + GUI_SIZE, y + GUI_SIZE, fill=color, outline="black"
                )

    def reset_episode(self):
        self.state = START
        self.ep = getattr(self, "ep", 0)
        self.canvas.coords(self.agent_vis, *self.cell_coords(self.state))
        self.canvas.itemconfig(self.info_id, text=f"Ep {self.ep+1}/{self.episodes}")

    def cell_coords(self, state):
        i, j = state
        x, y = j * GUI_SIZE, i * GUI_SIZE
        return (x + 5, y + 5, x + GUI_SIZE - 5, y + GUI_SIZE - 5)

    def train_step(self):
        if self.ep >= self.episodes:
            print("Treino concluído")
            self.demonstrar_resultados()
            return
        action = self.agent.choose_action(self.state)
        next_state = step(self.state, action)
        reward = R[next_state[0]][next_state[1]]
        self.agent.update(self.state, action, reward, next_state)
        self.state = next_state
        self.canvas.coords(self.agent_vis, *self.cell_coords(next_state))
        self.canvas.itemconfig(self.info_id, text=f"Ep {self.ep+1}/{self.episodes}")
        if self.state == GOAL:
            self.ep += 1
            self.reset_episode()
        self.root.after(UPDATE_INTERVAL, self.train_step)

    def demonstrar_resultados(self):
        arrow = {"N": "↑", "S": "↓", "L": "→", "O": "←"}

        for (i, j), qvals in self.agent.Q.items():
            best_idx = int(np.argmax(qvals))
            best_act = ACTIONS[best_idx]
            x = j * GUI_SIZE + GUI_SIZE / 2
            y = i * GUI_SIZE + GUI_SIZE / 2
            self.canvas.create_text(
                x, y, text=arrow[best_act], font=("Arial", 12, "bold"), fill="blue"
            )


if __name__ == "__main__":
    agent = QLearningAgent(gamma=0.9)
    VisualApp(agent, episodes=EPISODES)
