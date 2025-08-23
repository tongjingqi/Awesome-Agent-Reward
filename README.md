# Awesome-Agent-Reward

A curated list of awesome resources for reward construction in AI agents. This repository covers seminal papers, cutting-edge research, and practical guides on shaping and defining rewards to build more intelligent and aligned autonomous agents.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Foundations](#theoretical-foundations)
- [Verifiable Task Construction](#verifiable-task-construction)
- [Converting Open-Domain Tasks to Verifiable Tasks](#converting-open-domain-tasks-to-verifiable-tasks)
- [Reward Model Construction](#reward-model-construction)
- [Real-World Task Reward Construction](#real-world-task-reward-construction)
- [Unsupervised Reward Construction](#unsupervised-reward-construction)
- [World Model & Real-World Reward Construction](#world-model--real-world-reward-construction)
- [Evaluation and Benchmarks](#evaluation-and-benchmarks)
- [Contributing](#contributing)

## Introduction

**What is Reward Construction?**

Reward construction is the process of designing and implementing reward signals that guide AI agents toward desired behaviors and outcomes. It bridges the gap between human intentions and machine learning objectives, enabling agents to learn complex behaviors through reinforcement learning.

**Why is Reward Construction Important?**

As we transition from the "first half" (exam-based evaluation) to the "second half" (project-based interaction) of AI development, reward construction becomes crucial for:
- Enabling agents to learn from environmental feedback
- Supporting continuous learning from interaction history  
- Developing superhuman planning and reasoning capabilities
- Moving beyond human-derived data limitations

## Theoretical Foundations

### The Second Half & Era of Experience

**The Second Half**: Transitioning from creating new methods and models to defining new tasks
- **First Half Focus**: Exam-like tasks with universal methods (next token prediction, RL) and architectures (Transformer, GPT)
- **Turning Point**: Organic combination of universal methods and architectures, where RL on large models achieves generalization
- **Second Half Focus**: Project-based scenarios with multi-turn interactions and temporal learning

**Era of Experience**: Large Models + Reinforcement Learning = General Superhuman Agents
- **Previous Era**: Human Data Era with limitations of human-generated data and capabilities
- **Current Opportunity**: Combining self-discovery capabilities with task generality from the human data era
- **Key Components**: Environmental rewards, autonomous interaction, continuous experience streams, non-human planning and reasoning

## Verifiable Task Construction

Scaling task quantities through constructing verifiable task gyms, enhancing model general capabilities by solving puzzles and games.

### Multi-Modal Reasoning
- **Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning** - Using game code to synthesize verifiable multi-modal reasoning data for improving VLM general reasoning through RL

- **Play to Generalize: Learning to Reason Through Game Play** - Training expert models on Snake game to provide ground truth for multi-modal reasoning rewards, improving general reasoning capabilities through RL

### Zero-Sum Games & Strategic Reasoning  
- **SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning** - Using chess games' natural verifiable environments for self-play with win/loss rewards to enhance general reasoning

### Text-Based Puzzle Solving
- **InterThinker** - Advanced reasoning through structured thinking processes

- **SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond** - Systematic approach to generating logical reasoning data

- **Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles** - Creating synthetic puzzles to enhance logical reasoning capabilities

## Converting Open-Domain Tasks to Verifiable Tasks

Transforming next token prediction and pre-training tasks into RL-compatible formats.

- **Reinforcement Pre-training** - Converting next token prediction tasks into verifiable rewards where the next token serves as the verification signal

- **Learning to Reason for Long-Form Story Generation** - Constructing next chapter prediction tasks for story generation, using subsequent chapters as ground truth for reward construction

- **RLPR: Extrapolating RLVR to General Domains without Verifiers** - Extending reinforcement learning from verifiable reasoning to general domains

- **Reinforcing General Reasoning without Verifiers** - General reasoning enhancement without explicit verification mechanisms

## Reward Model Construction

Training reward models from preference data to enable policy learning on general tasks.

### Preference Modeling & Scaling
- **WorldPM: Scaling Human Preference Modeling** - Exploring scalability of preference modeling using 15M human forum data, showing clear scaling laws in objective tasks like mathematics

### Multi-Modal & Specialized Reward Models  
- **WavReward: Spoken Dialogue Models With Generalist Reward Evaluators** - Reward evaluation for spoken dialogue systems

- **VLM-based Reward Construction for Embodied Intelligence** - Using Vision-Language Models to construct rewards for embodied AI tasks

### Process Supervision
- **Process Reward Modeling** - Supervising intermediate reasoning steps rather than just final outcomes

## Real-World Task Reward Construction

Directly targeting real-world applications with practical reward construction.

- **WebDancer: Towards Autonomous Information Seeking Agency** - Synthesizing agent action trajectories for search tasks, constructing verifiable QA pairs for RL training on information seeking

## Unsupervised Reward Construction

Finding reward signals from model internals without external supervision.

- **Absolute Zero: Reinforced Self-play Reasoning with Zero Data** - Models serving dual roles as problem proposers and solvers, using compiler verification for self-improvement without external data

- **Enhancing Reasoning Ability through RL without Labels** - Completely unsupervised methods for reasoning enhancement

- **The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models** - Understanding how entropy affects reinforcement learning in reasoning models

## World Model & Real-World Reward Construction

Using world models and real-world interactions for reward construction.

- **Can Language Models Serve as Text-Based World Simulators?** - Exploring LLMs as world simulators for reward construction

- **Embodied Intelligence Reward Construction** - Comprehensive approaches to reward design in embodied AI systems

## Evaluation and Benchmarks

### Reward Model Evaluation
- **RewardBench** - Comprehensive benchmark for reward model evaluation

- **VL-RewardBench** - Vision-language reward model benchmarking

### Learning Capability Assessment  
- **EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving** - Evaluating learning capabilities through sequential problem solving rather than independent test cases

## Comparison of Reward Construction Approaches

| Approach | Generality & Transfer | Task Type | Implementation Difficulty |
|----------|----------------------|-----------|--------------------------|
| **Verifiable Task Construction** | Moderate - Puzzle/game solving as AGI prerequisite | Toy tasks with indirect real-world transfer | Medium - Requires manual task scaling |
| **Open-Domain → Verifiable** | High generality, transfer depends on task type | Medium toy level - Models general tasks | Low manual effort - Natural data sources with built-in rewards |
| **Reward Model Based** | Strong neural network generality but hackable | Real tasks | Depends on data collection difficulty |
| **Real-World Task** | Depends on task coverage | Direct real-world applications | Moderate - Automated but needs diversity consideration |
| **Unsupervised Internal** | Relatively strong | Both toy and real tasks | Low human effort - Potential for superhuman emergence |
| **World Model & Real-World** | Strong - Depends on world model generality | Varies from toy to real | Higher difficulty for more general approaches |

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
  author={[Jingqi Tong, Yurong Mou, Hangcheng li, Jun zhao]},
  year={2025},
  url={https://github.com/tongjingqi/Awesome-Agent-Reward}
}
```

---

**Note**: This is a living document that will be continuously updated as the field of agent reward construction evolves. Stay tuned for the latest developments!
