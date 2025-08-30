![image](top.png)

# Awesome-Agent-Reward-Construction

A curated list of awesome resources about reward construction for AI agents. This repository covers cutting-edge research, and practical guides on defining and collecting rewards to build more intelligent and aligned AI agents.

## Table of Contents

- [Introduction](#introduction)
- [Synthesizing Verifiable Task](#verifiable-task-construction)
- [Real-World Task Reward Construction](#real-world-task-reward-construction)
- [Unsupervised Reward Construction](#unsupervised-reward-construction)
- [Reward Model](#reward-model)
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
We divided Reward Construction research into 5 categories, including **Synthesizing Verifiable Task**, **Real-World Task Reward Construction**, **Unsupervised Reward Construction**, **Reward Model** and **Evaluation and Benchmarks**.

## Synthesizing Verifiable Task

Scaling task quantities through constructing new verifiable task gyms, such as puzzles, games. Training agents to solve these task can enhance model general reasoning capabilities. We divided Reward Construction research to 4 types, including **Multi-Modal Reasoning**, **Text-Based Puzzle Solving**, **Zero-Sum Games**, **Converting Open-Domain Tasks to Verifiable Tasks** and **Curriculum Learning**.

### Multi-Modal Reasoning
- [2505] [![Stargazers](https://img.shields.io/github/stars/tongjingqi/Code2Logic?style=social)](https://github.com/tongjingqi/Code2Logic) [Code2Logic: Game-Code-Driven Data Synthesis for Enhancing VLMs General Reasoning (Fudan University, Douyin Co., Ltd., Shanghai Innovation Institute, Shanghai Key Laboratory of Intelligent Information Processing, Institute of Modern Languages, Fudan University, Southern University of Science and Technology)](https://arxiv.org/abs/2505.13886) - Using game code to synthesize verifiable multi-modal reasoning data for improving VLM general reasoning through RL

- [2506] [![Stargazers](https://img.shields.io/github/stars/yunfeixie233/ViGaL?style=social)](https://github.com/yunfeixie233/ViGaL) [Play to Generalize: Learning to Reason Through Game Play (Rice University, Johns Hopkins University, NVIDIA)](https://arxiv.org/abs/2506.08011) - Training expert models on Snake game to provide ground truth for multi-modal reasoning rewards, improving general reasoning capabilities through RL


### Text-Based Puzzle Solving
- [2508] [![Stargazers](https://img.shields.io/github/stars/InternLM/InternBootcamp?style=social)](https://github.com/InternLM/InternBootcamp) [InternBootcamp Technical Report: Boosting LLM Reasoning with Verifiable Task Scaling (Shanghai AI Laboratory, Fudan University)](https://arxiv.org/pdf/2508.08636) - Advanced reasoning through structured thinking processes

- [2505] [![Stargazers](https://img.shields.io/github/stars/MiniMax-AI/SynLogic?style=social)](https://github.com/MiniMax-AI/SynLogic) [SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond (The Hong Kong University of Science and Technology, MiniMax, The City University of Hong Kong)](https://arxiv.org/abs/2505.19641) - Systematic approach to generating logical reasoning data

- [2505] [![Stargazers](https://img.shields.io/github/stars/BytedTsinghua-SIA/Enigmata?style=social)](https://github.com/BytedTsinghua-SIA/Enigmata) [Enigmata: Scaling Logical Reasoning in Large Language Models with Synthetic Verifiable Puzzles (ByteDance Seed, Fudan University, Institute for AI Industry Research (AIR), Tsinghua University, Nanjing University, Shanghai Jiao Tong University, SIA-Lab of Tsinghua AIR and ByteDance Seed)](https://arxiv.org/abs/2505.19914) - Creating synthetic puzzles to enhance logical reasoning capabilities

### Zero-Sum Games

- [2506] [![Stargazers](https://img.shields.io/github/stars/spiral-rl/spiral?style=social)](https://github.com/spiral-rl/spiral) [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning (National University of Singapore, Centre for Frontier AI Research (CFAR), A*STAR, Northeastern University, Sea AI Lab, Plastic Labs, University of Washington)](https://arxiv.org/abs/2506.24119) - Using chess games' natural verifiable environments for self-play with win/loss rewards to enhance general reasoning

### Converting General Tasks to Verifiable Tasks

Transforming general tasks which usually trained through Pretraining and SFT, into RL-compatible formats.

- [2506] [Reinforcement Pre-training (Microsoft Research, Peking University, Tsinghua University)](https://arxiv.org/abs/2506.08007) - Converting next token prediction tasks into verifiable rewards where the next token serves as the verification signal

- [2503] [![Stargazers](https://img.shields.io/github/stars/Alex-Gurung/ReasoningNCP?style=social)](https://github.com/Alex-Gurung/ReasoningNCP) [Learning to Reason for Long-Form Story Generation (University of Edinburgh)](https://arxiv.org/abs/2503.22828) - Constructing next chapter prediction tasks for story generation, using subsequent chapters as ground truth for reward construction

- [2506] [![Stargazers](https://img.shields.io/github/stars/OpenBMB/RLPR?style=social)](https://github.com/OpenBMB/RLPR) [RLPR: Extrapolating RLVR to General Domains without Verifiers (Tsinghua University, National University of Singapore, Shanghai Qi Zhi Institute, Harbin Institute of Technology, Beijing University of Posts and Telecommunications, University of Illinois Urbana-Champaign)](https://arxiv.org/abs/2506.18254) - Extending reinforcement learning from verifiable reasoning to general domains

- [2505] [![Stargazers](https://img.shields.io/github/stars/sail-sg/VeriFree?style=social)](https://github.com/sail-sg/VeriFree) [Reinforcing General Reasoning without Verifiers (Sea AI Lab, Singapore, University of Chinese Academy of Sciences, Institute of Automation, Chinese Academy of Sciences, National University of Singapore, University of Oxford, Renmin University of China)](https://arxiv.org/abs/2505.21493) - General reasoning enhancement without explicit verification mechanisms

### Curriculum Learning

Scaling difficuity of task through curriculum learning, converting sparse reward to dense reward.
- [2507] [Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling (University of Edinburgh, Fudan University, Alibaba Group, University of Amsterdam, Stepfun)](https://arxiv.org/abs/2507.01679)

- [2402] [![Stargazers](https://img.shields.io/github/stars/WooooDyy/LLM-Reverse-Curriculum-RL?style=social)](https://github.com/WooooDyy/LLM-Reverse-Curriculum-RL) [Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning (Fudan University, ByteDance Research)](https://arxiv.org/pdf/2402.05808)

- [2405] [![Stargazers](https://img.shields.io/github/stars/StoneT2000/rfcl?style=social)](https://github.com/StoneT2000/rfcl) [Reverse Forward Curriculum Learning for Extreme Sample and Demonstration Efficiency in Reinforcement Learning (University of California, San Diego)](https://arxiv.org/pdf/2405.03379)

- [2505] [![Stargazers](https://img.shields.io/github/stars/liumy2010/UFT?style=social)](https://github.com/liumy2010/UFT) [UFT: Unifying Supervised and Reinforcement Fine-Tuning (Massachusetts Institute of Technology)](https://arxiv.org/pdf/2505.16984)

## Real-World Task Reward Construction

Design reward function and synthesis data to scale up the quantities of the real-world reward. We divided Real-World Task Reward Construction research into 4 types, including **Web Search**, **GUI**, **VLA** and **World Model**.

### Web Search
- [2505] [![Stargazers](https://img.shields.io/github/stars/Alibaba-NLP/WebAgent?style=social)](https://github.com/Alibaba-NLP/WebAgent) [WebDancer: Towards Autonomous Information Seeking Agency (Tongyi Lab, Alibaba Group)](https://arxiv.org/abs/2505.22648) - Synthesizing agent action trajectories for search tasks, constructing verifiable QA pairs for RL training on information seeking
  
- [2507] [![Stargazers](https://img.shields.io/github/stars/Alibaba-NLP/WebAgent?style=social)](https://github.com/Alibaba-NLP/WebAgent) [WebSailor: Navigating Super-human Reasoning for Web Agent (Tongyi Lab, Alibaba Group)](https://arxiv.org/pdf/2507.02592)

- [2507] [![Stargazers](https://img.shields.io/github/stars/Alibaba-NLP/WebAgent?style=social)](https://github.com/Alibaba-NLP/WebAgent) [WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization (Tongyi Lab, Alibaba Group)](https://arxiv.org/pdf/2507.15061)

- [2508] [![Stargazers](https://img.shields.io/github/stars/Alibaba-NLP/WebAgent?style=social)](https://github.com/Alibaba-NLP/WebAgent) [WebWatcher: Breaking New Frontiers of Vision-Language Deep Research Agent (Tongyi Lab, Alibaba Group)](https://arxiv.org/pdf/2508.05748)
  
- [2503] [![Stargazers](https://img.shields.io/github/stars/PeterGriffinJin/Search-R1?style=social)](https://github.com/PeterGriffinJin/Search-R1) [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning (University of Illinois at Urbana-Champaign, University of Massachusetts Amherst, Google Cloud AI Research)](https://arxiv.org/abs/2503.09516)

- [2408] [![Stargazers](https://img.shields.io/github/stars/sentient-engineering/agent-q?style=social)](https://github.com/sentient-engineering/agent-q) [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents (The AGI Company (MultiOn), Stanford University)](https://arxiv.org/pdf/2408.07199) - Combination of DPO, MCTS, and process supervision for web navigation tasks with multi-modal reasoning rewards

- [2411] [![Stargazers](https://img.shields.io/github/stars/THUDM/WebRL?style=social)](https://github.com/THUDM/WebRL) [WebRL: Training LLM Web Agents via Self-Evolving Online Curriculum Reinforcement Learning (Tsinghua University, Zhipu AI)](https://arxiv.org/pdf/2411.02337)
  
- [2508] [![Stargazers](https://img.shields.io/github/stars/inclusionAI/ASearcher?style=social)](https://github.com/inclusionAI/ASearcher) [Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL (IIIS, Tsinghua University, Ant Research, RL Lab, University of Washington)](https://arxiv.org/pdf/2508.07976)
  
- [2504] [![Stargazers](https://img.shields.io/github/stars/RUC-NLPIR/WebThinker?style=social)](https://github.com/RUC-NLPIR/WebThinker) [WebThinker: Empowering Large Reasoning Models with Deep Research Capability (Renmin University of China, BAAI, Huawei Poisson Lab)](https://arxiv.org/pdf/2504.21776)

### GUI

- [2506] [![Stargazers](https://img.shields.io/github/stars/OpenBMB/AgentCPM-GUI?style=social)](https://github.com/OpenBMB/AgentCPM-GUI) [AgentCPM-GUI: Building Mobile-Use Agents with Reinforcement Fine-Tuning (Tsinghua University, Renmin University of China, ModelBest Inc.)](https://arxiv.org/pdf/2506.01391)

- [2505] [![Stargazers](https://img.shields.io/github/stars/dvlab-research/ARPO?style=social)](https://github.com/dvlab-research/ARPO) [ARPO:End-to-End Policy Optimization for GUI Agents with Experience Replay (The Chinese University of Hong Kong, SmartMore, Hong Kong University of Science and Technology)](https://arxiv.org/pdf/2505.16282)

- [2504] [![Stargazers](https://img.shields.io/github/stars/ritzz-ai/GUI-R1?style=social)](https://github.com/ritzz-ai/GUI-R1) [GUI-R1: A Generalist R1-Style Vision-Language Action Model For GUI Agents (Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences, University of Chinese Academy of Sciences, National University of Singapore)](https://arxiv.org/pdf/2504.10458)

- [2412] [![Stargazers](https://img.shields.io/github/stars/OS-Copilot/OS-Genesis?style=social)](https://github.com/OS-Copilot/OS-Genesis) [OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis (Shanghai AI Laboratory, The University of Hong Kong, Johns Hopkins University, Shanghai Jiao Tong University, University of Oxford, Hong Kong University of Science and Technology)](https://arxiv.org/pdf/2412.19723)

- [2506] [![Stargazers](https://img.shields.io/github/stars/sunblaze-ucb/AgentSynth?style=social)](https://github.com/sunblaze-ucb/AgentSynth) [AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents (UC Berkeley)](https://arxiv.org/abs/2506.14205)
  
### Embodied AI & Vision-Language-Action Model

- [2505] [![Stargazers](https://img.shields.io/github/stars/GuanxingLu/vlarl?style=social)](https://github.com/GuanxingLu/vlarl) [VLA-RL: Towards Masterful and General Robotic Manipulation with Scalable Reinforcement Learning (Tsinghua Shenzhen International Graduate School, Tsinghua University, School of Electrical and Electronic Engineering, Nanyang Technological University)](https://arxiv.org/abs/2505.18719) -  Reward design in embodied AI systems
  
- [2501] [Improving vision-language-action model with online reinforcement learning (Institute for Interdisciplinary Information Sciences, Tsinghua University, Shanghai Qi Zhi Institute, University of California, Berkeley)](https://arxiv.org/pdf/2501.16664)

- [2505] [![Stargazers](https://img.shields.io/github/stars/gen-robot/RL4VLA?style=social)](https://github.com/gen-robot/RL4VLA) [What Can RL Bring to VLA Generalization? (Tsinghua University, Shanghai Qi Zhi Institute, Beijing Zhongguancun Academy)](https://arxiv.org/pdf/2505.19789)

- [2505] [![Stargazers](https://img.shields.io/github/stars/Ariostgx/ript-vla?style=social)](https://github.com/Ariostgx/ript-vla) [Interactive Post-Training for Vision-Language-Action Models (UT Austin, Nankai University)](https://arxiv.org/pdf/2505.17016)

### World Model 
Towards future: Using world models and real-world interactions for reward construction.

- [2508] [Genie 3: A new frontier for world models (Google DeepMind)](https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/)
  
- [2508] [![Stargazers](https://img.shields.io/github/stars/SkyworkAI/Matrix-Game?style=social)](https://github.com/SkyworkAI/Matrix-Game) [Matrix-Game 2.0: An Open-Source, Real-Time, andStreaming Interactive World Model (Skywork AI)](https://arxiv.org/pdf/2508.13009)

- [2406] [![Stargazers](https://img.shields.io/github/stars/cognitiveailab/GPT-simulator?style=social)](https://github.com/cognitiveailab/GPT-simulator) [Can Language Models Serve as Text-Based World Simulators? (University of Arizona, New York University, Johns Hopkins University, Microsoft Research Montréal, Allen Institute for AI)](https://arxiv.org/abs/2406.06485) - Exploring LLMs as world simulators for reward construction

## Unsupervised Reward Construction

Finding reward signals from model internals. Model generates data to train itself. We divided Unsupervised Reward Construction into 2 types, including Proposer and Solver and the discussion of can Large Reasoning Models Self-Train.

### Proposer and Solver 

Models simultaneously act as problem proposers and solution generators, creating new training data.

- [2505] [Absolute Zero: Reinforced Self-play Reasoning with Zero Data (Tsinghua University, Beijing Institute for General Artificial Intelligence, Pennsylvania State University)](https://arxiv.org/abs/2505.03335) - Models serving dual roles as problem proposers and solvers, using compiler verification for self-improvement without external data

- [2508] [R-Zero: Self-Evolving Reasoning LLM from Zero Data (Tencent AI Seattle Lab, Washington University in St. Louis, University of Maryland, College Park, The University of Texas at Dallas)](https://arxiv.org/abs/2508.05004)

- [2508] [Self-Questioning Language Models (Carnegie Mellon University)](https://arxiv.org/pdf/2508.03682)

### Internal Signal Mining
Extracting learning signals from model internals, confidence scores, and consistant behaviors without external validation or verification mechanisms.

- [2505] [![Stargazers](https://img.shields.io/github/stars/PRIME-RL/Entropy-Mechanism-of-RL?style=social)](https://github.com/PRIME-RL/Entropy-Mechanism-of-RL) [The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models (Shanghai AI Laboratory, Tsinghua University, UIUC, Peking University, Nanjing University, CUHK)](https://arxiv.org/abs/2505.22617) - Understanding how entropy affects reinforcement learning in reasoning models, it indicated that the improvement of self-train may cause by the entropy mechanism.

- [2506] [![Stargazers](https://img.shields.io/github/stars/ruixin31/Spurious_Rewards?style=social)](https://github.com/ruixin31/Spurious_Rewards) [Spurious Rewards: Rethinking Training Signals in RLVR (University of Washington, Allen Institute for Artificial Intelligence, University of California, Berkeley)](https://arxiv.org/abs/2506.10947) - Can random reward improve model performance?

- [2505] [![Stargazers](https://img.shields.io/github/stars/zitian-gao/one-shot-em?style=social)](https://github.com/zitian-gao/one-shot-em) [Can Large Reasoning Models Self-Train (Ubiquant)](https://arxiv.org/abs/2505.20282)

- [2504] [![Stargazers](https://img.shields.io/github/stars/PRIME-RL/TTRL?style=social)](https://github.com/PRIME-RL/TTRL) [TTRL: Test-Time Reinforcement Learning (Tsinghua University, Shanghai AI Lab)](https://arxiv.org/abs/2504.16084)

- [2505] [![Stargazers](https://img.shields.io/github/stars/sail-sg/VeriFree?style=social)](https://github.com/sail-sg/VeriFree) [Enhancing Reasoning Ability through RL without Labels (Sea AI Lab, Singapore, University of Chinese Academy of Sciences, Institute of Automation, Chinese Academy of Sciences, National University of Singapore, University of Oxford, Renmin University of China)](https://arxiv.org/abs/2505.21493) - Completely unsupervised methods for reasoning enhancement

- [2505] [![Stargazers](https://img.shields.io/github/stars/sunblaze-ucb/Intuitor?style=social)](https://github.com/sunblaze-ucb/Intuitor) [Maximizing Confidence Alone Improves Reasoning (Carnegie Mellon University)](https://arxiv.org/abs/2505.22660)

- [2505] [![Stargazers](https://img.shields.io/github/stars/sail-sg/VeriFree?style=social)](https://github.com/sail-sg/VeriFree) [VeriFree: Reinforcing General Reasoning without Verifiers (Sea AI Lab, Singapore, University of Chinese Academy of Sciences, Institute of Automation, Chinese Academy of Sciences, National University of Singapore, University of Oxford, Renmin University of China)](https://arxiv.org/abs/2505.21493)

- [2504] [![Stargazers](https://img.shields.io/github/stars/QingyangZhang/EMPO?style=social)](https://github.com/QingyangZhang/EMPO) [Right Question is Already Half the Answer: Fully Unsupervised LLM Reasoning Incentivization (Tianjin University, Tencent AI Lab, NUS)](https://arxiv.org/abs/2504.05812)

- [2505] [![Stargazers](https://img.shields.io/github/stars/sunblaze-ucb/Intuitor?style=social)](https://github.com/sunblaze-ucb/Intuitor) [Learning to Reason without External Rewards (UC Berkeley, Yale University)](https://arxiv.org/abs/2505.19590)
  
## Reward Model Construction

Scaling preference data for reward models training to enable policy learning on general tasks.

### Generative Reward Model
- [2504] [Inference-Time Scaling for Generalist Reward Modeling (DeepSeek-AI, Dept. of Computer Sci. & Tech., Tsinghua University, Institute for AI Industry Research (AIR), Tsinghua University)](https://arxiv.org/abs/2504.02495)
  
- [2410] [Generative Reward Models (SynthLabs, Stanford University)](https://arxiv.org/pdf/2410.12832)

- [2408] [Generative Verifiers: Reward Modeling as Next-Token Prediction (Google DeepMind, University of Toronto, Mila, UCLA, Carnegie Mellon University)](https://arxiv.org/pdf/2408.15240)

### Reward Model Pretrain
- [2505] [![Stargazers](https://img.shields.io/github/stars/QwenLM/WorldPM?style=social)](https://github.com/QwenLM/WorldPM) [WorldPM: Scaling Human Preference Modeling (Qwen Team, Alibaba Group, Fudan University)](https://arxiv.org/abs/2505.10527) - Exploring scalability of preference modeling using 15M human forum data, showing clear scaling laws in objective tasks like mathematics
  
- [2507] [![Stargazers](https://img.shields.io/github/stars/InternLM/POLAR?style=social)](https://github.com/InternLM/POLAR) [POLAR：Pre-Trained Policy Discriminators are General Reward Models (Shanghai AI Laboratory, Fudan University)](https://arxiv.org/pdf/2507.05197)
  
### Multi-Modal Reward Models  
- [2505] [![Stargazers](https://img.shields.io/github/stars/jishengpeng/WavReward?style=social)](https://github.com/jishengpeng/WavReward) [WavReward: Spoken Dialogue Models With Generalist Reward Evaluators (Zhejiang University, Alibaba Group)](https://arxiv.org/abs/2505.09558) - Reward evaluation for spoken dialogue systems
  
- [2312] [Vision-Language Models as a Source of Rewards (Google DeepMind)](https://arxiv.org/abs/2312.09187)
  
- [2402] [Code as Reward: Empowering Reinforcement Learning with VLMs (Mila, McGill University, Google DeepMind, University of California, Berkeley)](https://arxiv.org/abs/2402.04764)
  
- [2402] [![Stargazers](https://img.shields.io/github/stars/yufeiwang63/RL-VLM-F?style=social)](https://github.com/yufeiwang63/RL-VLM-F) [RL-VLM-F: Reinforcement Learning from Vision Language Foundation Model Feedback (Robotics Institute, Carnegie Mellon University, Department of Computer Science, University of Southern California)](https://arxiv.org/abs/2402.03681)
  

### Process Supervision

- [2507] [Dynamic and Generalizable Process Reward Modeling (College of Computer Science and Artificial Intelligence, Fudan University, The University of Hong Kong)](https://arxiv.org/pdf/2507.17849)
  
- [2501] [The Lessons of Developing Process Reward Models in Mathematical Reasoning (Alibaba Group)](https://arxiv.org/abs/2501.07301) - Supervising intermediate reasoning steps rather than just final outcomes
  
- [2312] [![Stargazers](https://img.shields.io/github/stars/openreasoner/openr?style=social)](https://github.com/openreasoner/openr) [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations (National Key Laboratory for Multimedia Information Processing, Peking University, DeepSeek-AI, The University of Hong Kong, Tsinghua University, The Ohio State University)](https://arxiv.org/abs/2312.08935)
  
- [2305] [![Stargazers](https://img.shields.io/github/stars/openai/prm800k?style=social)](https://github.com/openai/prm800k) [Let’s Verify Step by Step (OpenAI)](https://arxiv.org/pdf/2305.20050)

## Evaluation and Benchmarks

Providing benchmarks or gyms to evaluate the model proformance. We divided Evaluation and Benchmarks into 4 types: Reward Model Benchmarks, Game Gym, Web Search, Computer Use and New Evaluation Dimension.

### Reward Model Benchmarks
- [2403] [![Stargazers](https://img.shields.io/github/stars/allenai/reward-bench?style=social)](https://github.com/allenai/reward-bench) [RewardBench (Allen Institute for Artificial Intelligence, University of Washington, Berkman Klein Center, Harvard Law)](https://arxiv.org/abs/2403.13787) - Comprehensive benchmark for reward model evaluation

- [2411] [![Stargazers](https://img.shields.io/github/stars/vl-rewardbench/VL_RewardBench?style=social)](https://github.com/vl-rewardbench/VL_RewardBench) [VL-RewardBench (HKU, SCUT, SJTU, PKU, UW, Allen AI)](https://arxiv.org/abs/2411.17451) - Vision-language reward model benchmarking

- [2501] [![Stargazers](https://img.shields.io/github/stars/ssmisya/PRMBench?style=social)](https://github.com/ssmisya/PRMBench) [PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward Models (Fudan University, Shanghai AI Laboratory, Soochow University, Stony Brook University, The Chinese University of Hong Kong)](https://arxiv.org/pdf/2501.03124)

- [2412] [![Stargazers](https://img.shields.io/github/stars/QwenLM/ProcessBench?style=social)](https://github.com/QwenLM/ProcessBench) [PROCESSBENCH (Alibaba Inc.)](https://arxiv.org/pdf/2412.06559)

### Game Gym
  
- [2402] [![Stargazers](https://img.shields.io/github/stars/jinhaoduan/GTBench?style=social)](https://github.com/jinhaoduan/GTBench) [GTBench: Uncovering the strategic reasoning limitations of llms via game-theoretic evaluations (Drexel University, Boston University, LLNL, Lehigh University, UNC Chapel Hill, MIT, Harvard University)](https://arxiv.org/abs/2402.12348)
  
- [2412] [![Stargazers](https://img.shields.io/github/stars/lmgame-org/GamingAgent?style=social)](https://github.com/lmgame-org/GamingAgent) [GameArena: Evaluating LLM Reasoning through Live Computer Games (University of California, San Diego, University of California, Berkeley)](https://arxiv.org/abs/2412.06394)
  
- [2403] [![Stargazers](https://img.shields.io/github/stars/CUHK-ARISE/GAMABench?style=social)](https://github.com/CUHK-ARISE/GAMABench) [How Far Are We on the Decision-Making of LLMs? Evaluating LLMs’ Gaming Ability in Multi-Agent Environments (The Chinese University of Hong Kong, Tencent AI Lab, The Chinese University of Hong Kong, Shenzhen, Tsinghua University)](https://arxiv.org/abs/2403.11807)
  
- [2411] [![Stargazers](https://img.shields.io/github/stars/balrog-ai/BALROG?style=social)](https://github.com/balrog-ai/BALROG) [Balrog: Benchmarking agentic llm and vlm reasoning on games (AI Centre, University College London, IDEAS NCBR, University of Warsaw, University of Oxford, New York University, Anthropic, Institute of Mathematics, Polish Academy of Sciences)](https://arxiv.org/abs/2411.13543)

- [2505] [![Stargazers](https://img.shields.io/github/stars/OS-Copilot/ScienceBoard?style=social)](https://github.com/OS-Copilot/ScienceBoard) [ScienceBoard: Evaluating Multimodal Autonomous Agents in Realistic Scientific Workflows (The University of Hong Kong, Shanghai AI Laboratory, Fudan University, Peking University, Nanjing University, East China Normal University, Yale University)](https://arxiv.org/pdf/2505.19897)

- [2406] [![Stargazers](https://img.shields.io/github/stars/Joshuaclymer/GameBench?style=social)](https://github.com/Joshuaclymer/GameBench) [GameBench: Evaluating strategic reasoning abilities of llm agents (Olin College of Engineering, University of Texas at Dallas, University of Oxford, University College London, ShanghaiTech University, Columbia University, University of Manchester)](https://arxiv.org/abs/2406.06613)

### Web Search Evaluation

- [2311] [GAIA: a benchmark for General AI Assistants (FAIR, Meta, HuggingFace, AutoGPT, GenAI, Meta)](https://arxiv.org/abs/2311.12983)
  
- [2504] [![Stargazers](https://img.shields.io/github/stars/openai/simple-evals?style=social)](https://github.com/openai/simple-evals) [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents (OpenAI)](https://arxiv.org/abs/2504.12516)
  
- [2504] [![Stargazers](https://img.shields.io/github/stars/PALIN2018/BrowseComp-ZH?style=social)](https://github.com/PALIN2018/BrowseComp-ZH) [BrowseComp-ZH: Benchmarking Web Browsing Ability of Large Language Models in Chinese (Hong Kong University of Science and Technology (Guangzhou), Peking University, Mindverse AI, Alibaba Group, Zhejiang University, Zhejiang University of Technology, MBZUAI, NIO, HSBC, Harvard T.H. Chan School of Public Health)](https://arxiv.org/abs/2504.19314)
  
### Computer Use Evaluation
- [2406] [![Stargazers](https://img.shields.io/github/stars/sierra-research/tau-bench?style=social)](https://github.com/sierra-research/tau-bench) [τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains (Mila – Quebec AI Research Institute, Polytechnique Montréal)](https://arxiv.org/pdf/2406.12045)

- [2404] [![Stargazers](https://img.shields.io/github/stars/xlang-ai/OSWorld?style=social)](https://github.com/xlang-ai/OSWorld) [OSWORLD: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments (The University of Hong Kong, CMU, Salesforce Research, University of Waterloo)](https://arxiv.org/pdf/2404.07972)
  
- [2412] [![Stargazers](https://img.shields.io/github/stars/TheAgentCompany/TheAgentCompany?style=social)](https://github.com/TheAgentCompany/TheAgentCompany) [TheAgentCompany: Benchmarking LLM Agents on Consequential Real World Tasks (Carnegie Mellon University, Duke University)](https://arxiv.org/abs/2412.14161)
  
- [2403] [![Stargazers](https://img.shields.io/github/stars/ServiceNow/WorkArena?style=social)](https://github.com/ServiceNow/WorkArena) [WorkArena: How Capable Are Web Agents at Solving Common Knowledge Work Tasks? (ServiceNow Research, Mila – Quebec AI Research Institute, Polytechnique Montréal, McGill University, Université de Montréal)](https://arxiv.org/abs/2403.07718)

- [2504] [![Stargazers](https://img.shields.io/github/stars/openai/simple-evals?style=social)](https://github.com/openai/simple-evals) [BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents (OpenAI)](https://arxiv.org/abs/2504.12516)


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
  author={[Jingqi Tong, Yurong Mou, Jun Zhao, Hangcheng Li, Yongzhuo Yang, Mingzhe Li, Zhangye Yin]},
  year={2025},
  url={https://github.com/tongjingqi/Awesome-Agent-Reward}
}
```

---

**Note**: This is a living document that will be continuously updated as the field of agent reward construction evolves. Stay tuned for the latest developments!
