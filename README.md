# Awesome-Agent-Reward

A curated list of awesome resources for **reward construction** in AI agents. This repository covers seminal papers, cutting-edge research, and practical guides on shaping and defining rewards to build more intelligent and aligned autonomous agents.

## Why Reward Construction?

We are moving from an era of "exams" (static benchmarks) to one of "projects" (interactive, multi-turn problem-solving). In this new paradigm, an agent's ability to learn from experience in its environment is paramount. This is the core idea behind concepts like **"The Second Half"** and the **"Era of Experience"**. The key to unlocking this potential lies in effectively defining and constructing reward signals.

By building robust reward mechanisms, we can guide agents to interact with their environment, learn from their own history, and develop reasoning and planning capabilities that may eventually surpass human-generated data. This repository categorizes and explores the diverse methodologies for reward construction.

## Papers

### 1\. Reward from Verifiable Environments (Games, Puzzles, etc.)

This approach uses tasks with clear, objective success criteria (like games or logical puzzles) to generate reward signals. It's a foundational method for enhancing the general reasoning capabilities of models.

  * **Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning**: Leverages the clear state transitions and causal logic inherent in game code to generate verifiable multi-modal reasoning data, which can be used to train more capable VLMs.
  * **Play to Generalize: Learning to Reason Through Game Play**: A small expert model is first trained to play a game (e.g., Snake). The expert's predictions then serve as ground-truth rewards to train a large multi-modal model, enhancing its general reasoning.
  * **SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning**: Uses the intrinsic win/loss signals in zero-sum games like chess for self-play. This allows an agent to improve its reasoning capabilities through reinforcement learning without requiring human-annotated data.
  * **InterThinker**: A framework for enhancing agent capabilities through interactive thinking and problem-solving in structured environments.

### 2\. Reward from Existing Data (Pre-training Data, Human Preferences)

These methods focus on extracting reward signals from large, pre-existing datasets, either by reframing pre-training objectives or by modeling human preferences.

#### 2.1 From Open-Domain Data

  * **Reinforcement Pre-training**: Frames the standard "next token prediction" task as a reinforcement learning problem, where the ground-truth next token serves as a verifiable reward signal, allowing RL to be applied to massive unsupervised datasets.
  * **Learning to Reason for Long-Form Story Generation**: Lacks a direct verifiable reward. This work constructs a reward by training a model to generate intermediate reasoning steps; a high reward is given if these steps increase the probability of generating the ground-truth next chapter.

#### 2.2 From Human Preferences (Reward Modeling)

  * **WorldPM: Scaling Human Preference Modeling**: Explores the scalability of preference modeling by training on 15 million examples of human interactions from online forums. It finds that scalability laws are most apparent in objective domains like mathematics.
  * **Process Reward Modeling**: Focuses on rewarding the *process* of reaching a solution, rather than just the final outcome. This encourages more robust and interpretable reasoning.

### 3\. Reward for Real-World Interaction

This category includes methods that construct rewards for agents operating in complex, real-world environments like the web.

  * **WebDancer: Towards Autonomous Information Seeking Agency**: Addresses the lack of agent trajectory data for search tasks. It first generates question-answer pairs on a target webpage and then uses a powerful model to distill the search and navigation trajectory, which is then used as a basis for reinforcement learning.

### 4\. Unsupervised & Self-Generated Rewards

A cutting-edge approach where agents learn with minimal or no human supervision by creating their own curriculum and reward signals.

  * **Absolute Zero: Reinforced Self-play Reasoning with Zero Data**: The model acts as both a "proposer" (creating problems) and a "solver." Since the tasks (e.g., coding) have a verifiable outcome via a compiler, the agent can use this feedback for RL, improving both its problem-generation and problem-solving skills without external data.
  * **Enhancing Reasoning Ability through RL without Labels**: Explores methods for improving reasoning through reinforcement learning signals derived internally from the model (e.g., entropy, exploration), rather than from external labels.

### 5\. Future Directions: World Models & Embodied Agents

This frontier explores using simulators and advanced models to generate rewards for agents that interact with the physical or simulated world.

  * **Can Language Models Serve as Text-Based World Simulators?**: Investigates the potential of LLMs to function as simulators for interactive environments, which could provide rich feedback for agent training.
  * **VLM for Embodied Intelligence Reward Construction**: Discusses how Vision-Language Models (VLMs) can be leveraged to create dense reward functions for robotics and embodied AI tasks, translating visual goals into tangible rewards.

## Contributing

Your contributions are always welcome\! If you have any papers or resources to add, please feel free to open a pull request.
