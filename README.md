# TIR-Judge Research Project

A research project investigating **Tool-Integrated Reinforcement Learning for LLM Judges** based on Google Research's breakthrough paper.

## 🎯 Project Overview

This project explores the TIR-Judge framework, which combines Large Language Model judges with code execution capabilities through reinforcement learning. The approach enables more accurate evaluation of responses by allowing judges to perform precise computation and verification.

## 📚 Research Focus

### Core Innovation: TIR-Judge
- **Tool Integration**: LLM judges augmented with Python code execution
- **Reinforcement Learning**: End-to-end training using DAPO (improved GRPO)
- **Self-Bootstrapping**: TIR-Judge-Zero learns without teacher distillation
- **Multi-Format Evaluation**: Supports pointwise, pairwise, and listwise judgment

### Key Findings from Original Research
- **6.4% improvement** in pointwise evaluation vs reasoning-based judges
- **7.7% improvement** in pairwise evaluation
- **96% performance** of Claude-Opus-4 with only 8B parameters
- **Self-improvement** without distillation through iterative RL

## 📁 Project Structure

```
./
├── docs/
│   ├── original-docs/          # Source papers and materials
│   │   ├── 2510.23038v1.pdf   # Original research paper
│   │   ├── 2510.23038v1.md    # Converted markdown
│   │   └── *.md               # Research notes
│   ├── analysis/               # Research analysis
│   └── summaries/              # Key findings
├── src/                        # Implementation code
├── data/                       # Datasets and experiments
├── results/                    # Experimental outputs
├── notebooks/                  # Jupyter analysis notebooks
├── scripts/                    # Automation scripts
├── CLAUDE.md                   # Project guide for Claude
└── README.md                   # This file
```

## 🔬 Research Areas

### 1. Tool-Integrated Reasoning (TIR)
- Code execution sandbox integration
- Multi-turn reasoning with tool feedback
- Verifiable constraint checking

### 2. Reinforcement Learning Framework
- Reward design (correctness + format + tool-use)
- Iterative training strategies
- Self-bootstrapping vs distillation

### 3. Evaluation Methodologies
- **Pointwise**: Individual response scoring
- **Pairwise**: Response comparison
- **Listwise**: Multi-response ranking

## 🚀 Getting Started

### Prerequisites
```bash
# Python environment
python >= 3.8
pytorch >= 1.9
transformers >= 4.20
```

### Quick Setup
```bash
# Clone and setup
git clone <repository-url>
cd tir-judge-research

# Install dependencies
pip install -r requirements.txt

# Explore the research
jupyter notebook notebooks/
```

## 📊 Key Results to Replicate

| Model | Params | PPE Pointwise | PPE Pairwise | RewardBench2 |
|-------|--------|---------------|--------------|--------------|
| TIR-Judge-Zero 8B | 8B | 67.8% | 76.6% | 73.4% |
| TIR-Judge-Distill 8B | 8B | 70.9% | 72.2% | 71.6% |
| Claude-Opus-4 | ~175B | - | - | 76.5% |

## 🎓 Research Questions

1. **Tool vs Text**: How does tool integration compare to pure text reasoning?
2. **Self-Learning**: Can judges bootstrap without teacher distillation?
3. **Scaling**: How does performance scale with model size and data?
4. **Generalization**: Do tool-augmented judges transfer across domains?

## 🔧 Implementation Roadmap

- [ ] **Phase 1**: Literature review and baseline implementation
- [ ] **Phase 2**: Core TIR-Judge framework development
- [ ] **Phase 3**: Experimental validation on benchmarks
- [ ] **Phase 4**: Ablation studies and analysis

## 📖 Documentation

- **Research Guide**: See [CLAUDE.md](./CLAUDE.md) for detailed project context
- **Original Paper**: Available in `docs/original-docs/2510.23038v1.pdf`
- **Analysis Notes**: Research findings in `docs/analysis/`

## 🤝 Contributing

This is a research project focused on understanding and extending TIR-Judge concepts:

1. Maintain research integrity and proper attribution
2. Follow systematic experimental practices
3. Document findings and methodologies
4. Use reproducible setups

## 📄 Citation

```bibtex
@article{xu2025tir,
  title={Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning},
  author={Xu, Ran and Chen, Jingjing and Ye, Jiayu and Wu, Yu and Yan, Jun and Yang, Carl and Yu, Hongkun},
  journal={arXiv preprint arXiv:2510.23038},
  year={2025}
}
```

## 🔗 Resources

- **Original Research**: Google Research & Emory University
- **Paper arXiv**: [2510.23038](https://arxiv.org/abs/2510.23038)
- **Related Work**: Tool-augmented LLMs, RL for NLP, LLM-as-a-judge

---

*A systematic exploration of tool-integrated reinforcement learning for more capable LLM judges.*