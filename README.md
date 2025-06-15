# RL_projects
MENACE Tic-Tac-Toe Reinforcement Learning

This project implements the MENACE (Machine Educable Noughts And Crosses Engine) algorithm for Tic-Tac-Toe using Python. MENACE is a classic reinforcement learning approach that uses matchboxes and colored beads to learn optimal strategies for playing Tic-Tac-Toe. This implementation simulates MENACE's learning process and allows for self-play, human play, and random policy play

### Features

- Full Tic-Tac-Toe environment and state management
- MENACE policy with reinforcement learning updates
- Human and random agent support
- Training and evaluation scripts
- Visualization of board states and learning process

### How to Use

1. Clone the repository and open the notebook in Google Colab.
2. Run all cells to train MENACE and play against it.
3. Modify parameters (like training epochs, temperature) as desired.

### Project Structure

- `State`, `Env`, and `Game` classes for environment and game logic
- `MenacePolicy`, `RandomPolicy`, and `HumanPolicy` for agent behavior
- Training and evaluation scripts

### References

- Donald Michie, "Trial and Error Learning in a Mechanism for Game-Playing," 1961
- [Wikipedia: MENACE (machine)](https://en.wikipedia.org/wiki/MENACE_(machine))
