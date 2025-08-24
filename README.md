# Awesome-Agent-Reward

A curated list of awesome resources for reward construction in AI agents. This repository covers seminal papers, cutting-edge research, and practical guides on shaping and defining rewards to build more intelligent and aligned autonomous agents.

## Table of Contents

- [Introduction](#introduction)
- [Verifiable Task Construction](#verifiable-task-construction)
- [Real-World Task Reward Construction](#real-world-task-reward-construction)
- [Unsupervised Reward Construction](#unsupervised-reward-construction)
- [Reward Model Construction](#reward-model-construction)
- [Evaluation and Benchmarks](#evaluation-and-benchmarks)
- [Contributing](#contributing)


## Introduction

**What is Reward Construction?**

Reward construction is the process of designing and collecting reward signals that guide AI agents toward desired behaviors and outcomes. 

**Why is Reward Construction Important?**

**Background: The Second Half & Era of Experience**

**The Second Half**: Transitioning from creating new methods and models to defining new tasks
- **First Half Focus**: Exam-like tasks with universal methods (next token prediction, RL) and architectures (Transformer, GPT)
- **Turning Point**: Organic combination of universal methods and architectures, where RL on large models achieves generalization
- **Second Half Focus**: Project-based scenarios with multi-turn interactions and temporal learning

**Era of Experience**: Large Models + Reinforcement Learning = General Superhuman Agents
- **Previous Era**: Human Data Era with limitations of human-generated data and capabilities
- **Current Opportunity**: Combining self-discovery capabilities with task generality from the human data era
- **Key Components**: Environmental rewards, autonomous interaction, continuous experience streams, non-human planning and reasoning

In conclusion, reward construction provides **interactive environments** and **learning signals**. It becomes crucial for AI agent to **get experience for new project**. 
We divided Reward Construction research into 5 categories, including **Verifiable Task Construction**, **Real-World Task Reward Construction**, **Unsupervised Reward Construction**, **Reward Model Construction** and **Evaluation and Benchmarks**.

## Verifiable Task Construction

Scaling task quantities through constructing new verifiable task gyms, such as puzzles and games, enhance model general capabilities.

### Multi-Modal Reasoning
- [2505] [Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning](https://arxiv.org/abs/2505.13886) - Using game code to synthesize verifiable multi-modal reasoning data for improving VLM general reasoning through RL

- [2506] [Play to Generalize: Learning to Reason Through Game Play](https://arxiv.org/abs/2506.08011) - Training expert models on Snake game to provide ground truth for multi-modal reasoning rewards, improving general reasoning capabilities through RL

### Text-Based Puzzle Solving
- [2508] [InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling](https://arxiv.org/pdf/2508.08636) - Advanced reasoning through structured thinking processes

- [2505] [SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond](https://arxiv.org/abs/2505.19641) - Systematic approach to generating logical reasoning data

- [2505] [Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles](https://arxiv.org/abs/2505.19914) - Creating synthetic puzzles to enhance logical reasoning capabilities

### Zero-Sum Games & Strategic Reasoning  
- [2506] [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2506.24119) - Using chess games' natural verifiable environments for self-play with win/loss rewards to enhance general reasoning
  
### Converting Open-Domain Tasks to Verifiable Tasks

Transforming next token prediction and pre-training tasks into RL-compatible formats.

- [2506] [Reinforcement Pre-training](https://arxiv.org/abs/2506.08007) - Converting next token prediction tasks into verifiable rewards where the next token serves as the verification signal

- [2503] [Learning to Reason for Long-Form Story Generation](https://arxiv.org/abs/2503.22828) - Constructing next chapter prediction tasks for story generation, using subsequent chapters as ground truth for reward construction

- [2506] [RLPR: Extrapolating RLVR to General Domains without Verifiers](https://arxiv.org/abs/2506.18254) - Extending reinforcement learning from verifiable reasoning to general domains

- [2505] [Reinforcing General Reasoning without Verifiers](https://arxiv.org/abs/2505.21493) - General reasoning enhancement without explicit verification mechanisms

### Sparse Reward

- [2101] [Asymmetric self-play for automatic goal discovery in robotic manipulation](https://arxiv.org/pdf/2101.04882)

- [2402] [Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning](https://arxiv.org/pdf/2402.05808)

- [2405] [Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in Reinforcement Learning](https://arxiv.org/pdf/2405.03379)

## Real-World Task Reward Construction

Directly targeting real-world applications with practical reward construction.

### Web Search
- [2505] [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648) - Synthesizing agent action trajectories for search tasks, constructing verifiable QA pairs for RL training on information seeking

- [2503] [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)

- [2408] [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/pdf/2408.07199) - Combination of DPO, MCTS, and process supervision for web navigation tasks with multi-modal reasoning rewards

- [2411] [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://arxiv.org/pdf/2411.02337)
  
- [2508] [Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL](https://arxiv.org/pdf/2508.07976)
  
- [2504] [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/pdf/2504.21776)

### GUI

- [2506] [AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning](https://arxiv.org/pdf/2506.01391)

- [2505] [ARPO:End-to-End Policy Optimization for GUI Agents with Experience Replay](https://arxiv.org/pdf/2505.16282)

- [2505] [GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents](https://arxiv.org/pdf/2505.15810)

- [2504] [GUI-R1: A Generalist R1-Style Vision-Language Action Model For GUI Agents](https://arxiv.org/pdf/2504.10458)

### VLA Reward Construction
- [2505] [VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning](https://arxiv.org/abs/2505.18719) -  Reward design in embodied AI systems

### World Model 
Towards future: Using world models and real-world interactions for reward construction.
- [2406] [Can Language Models Serve as Text-Based World Simulators?](https://arxiv.org/abs/2406.06485) - Exploring LLMs as world simulators for reward construction

## Unsupervised Reward Construction

Finding reward signals from model internals without external supervision.

- [2505] [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) - Models serving dual roles as problem proposers and solvers, using compiler verification for self-improvement without external data

- [2505] [Enhancing Reasoning Ability through RL without Labels](https://arxiv.org/abs/2505.21493) - Completely unsupervised methods for reasoning enhancement

- [2505] [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617) - Understanding how entropy affects reinforcement learning in reasoning models

## Reward Model Construction

Training reward models from preference data to enable policy learning on general tasks.

### Preference Modeling & Scaling
- [2505] [WorldPM: Scaling Human Preference Modeling](https://arxiv.org/abs/2505.10527) - Exploring scalability of preference modeling using 15M human forum data, showing clear scaling laws in objective tasks like mathematics

### Multi-Modal & Specialized Reward Models  
- [2505] [WavReward: Spoken Dialogue Models With Generalist Reward Evaluators](https://arxiv.org/abs/2505.09558) - Reward evaluation for spoken dialogue systems

### Process Supervision
- [2501] [Process Reward Modeling](https://arxiv.org/abs/2501.07301) - Supervising intermediate reasoning steps rather than just final outcomes

## Evaluation and Benchmarks

### Reward Model Evaluation
- [2403] [RewardBench](https://arxiv.org/abs/2403.13787) - Comprehensive benchmark for reward model evaluation

- [2411] [VL-RewardBench](https://arxiv.org/abs/2411.17451) - Vision-language reward model benchmarking

### Learning Capability Assessment  
- [2506] [EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](https://arxiv.org/abs/2506.02672) - Evaluating learning capabilities through sequential problem solving rather than independent test cases


## Contributing

We welcome contributions to this repository! Please feel free to:

1. Submit pull requests to add new papers
2. Improve paper categorization and descriptions  
3. Add implementation details or code repositories
4. Suggest new categories or reorganization

When adding papers, please include:
- Paper title and authors
- Brief description of the reward construction method
- Key contributions and results
- Links to paper and code (if available)

## Citation

If you find this repository useful, please consider citing:

```bibtex
@misc{awesome-agent-reward,
  title={Awesome Agent Reward: Reward Construction for AI Agents},
  author={[Jingqi Tong, Yurong Mou, Hangcheng Li, Jun Zhao]},
  year={2025},
  url={https://github.com/tongjingqi/Awesome-Agent-Reward}
}
```

---

**Note**: This is a living document that will be continuously updated as the field of agent reward construction evolves. Stay tuned for the latest developments!
