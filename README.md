# Awesome-Agent-Reward-Construction

A curated list of awesome resources about reward construction for AI agents. This repository covers cutting-edge research, and practical guides on defining and collecting rewards to build more intelligent and aligned AI agents.

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

**[The Second Half](https://ysymyth.github.io/The-Second-Half/)**: Transitioning from creating new methods and models to defining new tasks
- **First Half Focus**: Exam-like tasks with universal methods (next token prediction, RL) and architectures (Transformer, GPT)
- **Turning Point**: Organic combination of universal methods and architectures, where RL on large models achieves generalization
- **Second Half Focus**: Project-based scenarios with multi-turn interactions and temporal learning

**[Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)**: Large Models + Reinforcement Learning = General Superhuman Agents
- **Previous Era**: Human Data Era with limitations of human-generated data and capabilities
- **Current Opportunity**: Combining self-discovery capabilities with task generality from the human data era
- **Key Components**: Environmental rewards, autonomous interaction, continuous experience streams, non-human planning and reasoning

In conclusion, reward construction provides **interactive environments** and **learning signals**. It becomes crucial for AI agent to **get experience for new project**. 
We divided Reward Construction research into 5 categories, including **Verifiable Task Construction**, **Real-World Task Reward Construction**, **Unsupervised Reward Construction**, **Reward Model Construction** and **Evaluation and Benchmarks**.

## Verifiable Task Construction

Scaling task quantities through constructing new verifiable task gyms, such as puzzles, games. It can enhancing model general capabilities. We divided Reward Construction research to 4 types, including **Multi-Modal Reasoning**, **Text-Based Puzzle Solving**, **Zero-Sum Games**, **Converting Open-Domain Tasks to Verifiable Tasks** and **Curriculum Learning**.

### Multi-Modal Reasoning
- [2505] [Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning](https://arxiv.org/abs/2505.13886) - Using game code to synthesize verifiable multi-modal reasoning data for improving VLM general reasoning through RL

- [2506] [Play to Generalize: Learning to Reason Through Game Play](https://arxiv.org/abs/2506.08011) - Training expert models on Snake game to provide ground truth for multi-modal reasoning rewards, improving general reasoning capabilities through RL


### Text-Based Puzzle Solving
- [2508] [InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling](https://arxiv.org/pdf/2508.08636) - Advanced reasoning through structured thinking processes

- [2505] [SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond](https://arxiv.org/abs/2505.19641) - Systematic approach to generating logical reasoning data

- [2505] [Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles](https://arxiv.org/abs/2505.19914) - Creating synthetic puzzles to enhance logical reasoning capabilities

### Zero-Sum Games
- [2506] [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2506.24119) - Using chess games' natural verifiable environments for self-play with win/loss rewards to enhance general reasoning

### Converting Open-Domain Tasks to Verifiable Tasks

Transforming next token prediction and pre-training tasks into RL-compatible formats.

- [2506] [Reinforcement Pre-training](https://arxiv.org/abs/2506.08007) - Converting next token prediction tasks into verifiable rewards where the next token serves as the verification signal

- [2503] [Learning to Reason for Long-Form Story Generation](https://arxiv.org/abs/2503.22828) - Constructing next chapter prediction tasks for story generation, using subsequent chapters as ground truth for reward construction

- [2506] [RLPR: Extrapolating RLVR to General Domains without Verifiers](https://arxiv.org/abs/2506.18254) - Extending reinforcement learning from verifiable reasoning to general domains

- [2505] [Reinforcing General Reasoning without Verifiers](https://arxiv.org/abs/2505.21493) - General reasoning enhancement without explicit verification mechanisms

- [2507] [KIMI K2: OPEN AGENTIC INTELLIGENCE](https://arxiv.org/pdf/2507.20534)

### Curriculum Learning

Scaling difficuity of task through curriculum learning, converting sparse reward to dense reward.
- [2507] [Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling](https://arxiv.org/abs/2507.01679)

- [2505] [UFT: Unifying Supervised and Reinforcement Fine-Tuning](https://arxiv.org/pdf/2505.16984)

- [2402] [Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning](https://arxiv.org/pdf/2402.05808)

- [2405] [Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in Reinforcement Learning](https://arxiv.org/pdf/2405.03379)

## Real-World Task Reward Construction

Design reward function and synthesis data to scale up the quantities of the real-world reward. We divided Real-World Task Reward Construction research into 4 types, including **Web Search**, **GUI**, **VLA** and **World Model**.

### Web Search
- [2505] [WebDancer: Towards Autonomous Information Seeking Agency](https://arxiv.org/abs/2505.22648) - Synthesizing agent action trajectories for search tasks, constructing verifiable QA pairs for RL training on information seeking

- [2503] [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)

- [2408] [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/pdf/2408.07199) - Combination of DPO, MCTS, and process supervision for web navigation tasks with multi-modal reasoning rewards

- [2411] [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning](https://arxiv.org/pdf/2411.02337)
  
- [2508] [Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL](https://arxiv.org/pdf/2508.07976)
  
- [2504] [WebThinker: Empowering Large Reasoning Models with Deep Research Capability](https://arxiv.org/pdf/2504.21776)

### GUI
- [2506] [AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](https://arxiv.org/abs/2506.14205)

- [2506] [AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning](https://arxiv.org/pdf/2506.01391)

- [2505] [ARPO:End-to-End Policy Optimization for GUI Agents with Experience Replay](https://arxiv.org/pdf/2505.16282)

- [2505] [GUI-G1: Understanding R1-Zero-Like Training for Visual Grounding in GUI Agents](https://arxiv.org/pdf/2505.15810)

- [2504] [GUI-R1: A Generalist R1-Style Vision-Language Action Model For GUI Agents](https://arxiv.org/pdf/2504.10458)

- [2412] [OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis](https://arxiv.org/pdf/2412.19723)

### Tool
- [2504] [ToolRL: Reward is All Tool Learning Needs](https://arxiv.org/abs/2504.13958)
  
- [2504] [ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/pdf/2504.11536)

- [2503] [TORL: Scaling Tool-Integrated RL](https://arxiv.org/abs/2503.23383)
  
### VLA
- [2505] [VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning](https://arxiv.org/abs/2505.18719) -  Reward design in embodied AI systems
  
- [2412] [RLDG: Robotic Generalist Policy Distillation via Reinforcement Learning](https://arxiv.org/pdf/2412.09858)

- [2501] [Improving vision-language-action model with online reinforcement learning](https://arxiv.org/pdf/2501.16664)

- [2505] [What Can RL Bring to VLA Generalization?](https://arxiv.org/pdf/2505.19789)

- [2505] [Interactive Post-Training for Vision-Language-Action Models](https://arxiv.org/pdf/2505.17016)

### World Model 
Towards future: Using world models and real-world interactions for reward construction.
- [2406] [Can Language Models Serve as Text-Based World Simulators?](https://arxiv.org/abs/2406.06485) - Exploring LLMs as world simulators for reward construction
- [2508] [Genie 3: A new frontier for world models](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)
- [2508] [Matrix-Game 2.0: An Open-Source, Real-Time, andStreaming Interactive World Model](https://arxiv.org/pdf/2508.13009)
## Unsupervised Reward Construction

Finding reward signals from model internals without external supervision. We divided Unsupervised Reward Construction into 2 types, including Proposer and Solver and the discussion of can Large Reasoning Models Self-Train.

### Proposer and Solver 

- [2505] [Absolute Zero: Reinforced Self-play Reasoning with Zero Data](https://arxiv.org/abs/2505.03335) - Models serving dual roles as problem proposers and solvers, using compiler verification for self-improvement without external data

- [2508] [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004)

- [2508] [Self-Questioning Language Models](https://arxiv.org/pdf/2508.03682)

### Can Large Reasoning Models Self-Train?

Discussion of can Large Reasoning Models Self-Train.

- [2505] [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models](https://arxiv.org/abs/2505.22617) - Understanding how entropy affects reinforcement learning in reasoning models, it indicated that the improvement of self-train may cause by the entropy mechanism.

- [2506] [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947) - Can random reward improve model performance?

- [2505] [Can Large Reasoning Models Self-Train](https://arxiv.org/abs/2505.20282)

- [2504] [TTRL: Test-Time Reinforcement Learning](https://arxiv.org/abs/2504.16084)

- [2505] [Enhancing Reasoning Ability through RL without Labels](https://arxiv.org/abs/2505.21493) - Completely unsupervised methods for reasoning enhancement

- [2505] [Maximizing Confidence Alone Improves Reasoning](https://arxiv.org/abs/2505.22660)

- [2505] [VeriFree: Reinforcing General Reasoning without Verifiers](https://arxiv.org/abs/2505.21493)

- [2504] [Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization](https://arxiv.org/abs/2504.05812)

- [2505] [Learning to Reason without External Rewards](https://arxiv.org/abs/2505.19590)
  
## Reward Model Construction

Scaling preference data for reward models training to enable policy learning on general tasks.

### Generative Reward Model
- [2504] [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495)
  
- [2410] [Generative Reward Models](https://arxiv.org/pdf/2410.12832)

- [2408] [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/pdf/2408.15240)

### Reward Model Pretrain
- [2505] [WorldPM: Scaling Human Preference Modeling](https://arxiv.org/abs/2505.10527) - Exploring scalability of preference modeling using 15M human forum data, showing clear scaling laws in objective tasks like mathematics
  
- [2507] [POLAR：Pre-Trained Policy Discriminators are General Reward Models](https://arxiv.org/pdf/2507.05197)
  
### Multi-Modal Reward Models  
- [2505] [WavReward: Spoken Dialogue Models With Generalist Reward Evaluators](https://arxiv.org/abs/2505.09558) - Reward evaluation for spoken dialogue systems
- [2312] [Vision-Language Models as a Source of Rewards](https://arxiv.org/abs/2312.09187)
- [2402] [Code as Reward: Empowering Reinforcement Learning with VLMs](https://arxiv.org/abs/2402.04764)
- [2402] [RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback](https://arxiv.org/abs/2402.03681)
  

### Process Supervision

- [2507] [Dynamic and Generalizable Process Reward Modeling](https://arxiv.org/pdf/2507.17849)
- [2501] [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301) - Supervising intermediate reasoning steps rather than just final outcomes
- [2312] [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)
- [2305] [Let’s Verify Step by Step](https://arxiv.org/pdf/2305.20050)

## Evaluation and Benchmarks

Providing benchmarks or gyms to evaluation the model proformance. We divided Evaluation and Benchmarks into 4 types: Reward Model Benchmarks, Game Gym, Web Search, Computer Use and New Evaluation Dimension.

### Reward Model Benchmarks
- [2403] [RewardBench](https://arxiv.org/abs/2403.13787) - Comprehensive benchmark for reward model evaluation

- [2411] [VL-RewardBench](https://arxiv.org/abs/2411.17451) - Vision-language reward model benchmarking

- [2501] [PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models](https://arxiv.org/pdf/2501.03124)

- [2412] [PROCESSBENCH](https://arxiv.org/pdf/2412.06559)

### Game Gym

- [2406] [GameBench: Evaluating strategic reasoning abilities of llm agents](https://arxiv.org/abs/2406.06613)
  
- [2402] [GTBench: Uncovering the strategic reasoning limitations of llms via game-theoretic evaluations](https://arxiv.org/abs/2402.12348)
  
- [2412] [GameArena: Evaluating LLM Reasoning through Live Computer Games](https://arxiv.org/abs/2412.06394)
  
- [2403] [How Far Are We on the Decision-Making of LLMs? Evaluating LLMs’ Gaming Ability in Multi-Agent Environments](https://arxiv.org/abs/2403.11807)
  
- [2411] [Balrog: Benchmarking agentic llm and vlm reasoning on games](https://arxiv.org/abs/2411.13543)
  
- [2503] [DSGBench: A diverse strategic game benchmark for evaluating llm-based agents in complex decision-making environments](https://arxiv.org/abs/2503.06047)
  
- [2410] [ING-VP: Mllms cannot play easy vision-based games yet](https://arxiv.org/abs/2410.06555)

- [2505] [ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows](https://arxiv.org/pdf/2505.19897)

### Web Search

### Computer Use

- [2404] [OSWORLD: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://arxiv.org/pdf/2404.07972)
  
- [2406] [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains](https://arxiv.org/pdf/2406.12045)
  
- [2412] [TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks](https://arxiv.org/abs/2412.14161)
  
- [2403] [WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks?](https://arxiv.org/abs/2403.07718)

- [2504] [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents](https://arxiv.org/abs/2504.12516)

### New dimension to evalate  
- [2506] [EvaLearn: Quantifying the Learning Capability and Efficiency of LLMs via Sequential Problem Solving](https://arxiv.org/abs/2506.02672) - Evaluating learning capabilities through sequential problem solving rather than independent test cases

## Limitation and Future Work

- Game data isn't utilized thoroughly. Games are shown effective to enhance model's general abilities, but open-source models rarely include games as training data.
- Using World Model to construct rewards. World models are capable of generating an endless variety of action-controllable, playable 3D environments for training and evaluating embodied agents.
- Evaluating interactable environments. A method to evaluate interactable environments, find high quality ones and choose environments fitting policy model's ability level would boost training a lot.


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
