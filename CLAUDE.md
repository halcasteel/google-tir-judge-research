# CLAUDE.md - TIR-Judge Research Project

## Project Overview

This research project focuses on analyzing and implementing concepts from "Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning" by Google Research. The project explores tool-augmented LLM judges that can execute code for more accurate evaluation.

## Research Context

### Primary Paper
- **Title**: Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning
- **Authors**: Ran Xu, Jingjing Chen, Jiayu Ye, Yu Wu, Jun Yan, Carl Yang, Hongkun Yu
- **Institution**: Google Research, Emory University
- **Key Innovation**: TIR-Judge framework combining reinforcement learning with code execution for precise LLM evaluation

### Research Goals
1. Understand tool-integrated reasoning (TIR) approaches for LLM judges
2. Analyze the effectiveness of reinforcement learning for training tool-augmented judges
3. Explore self-bootstrapping techniques (TIR-Judge-Zero) vs distillation approaches
4. Investigate multi-format evaluation (pointwise, pairwise, listwise)

## Project Structure

```
./
├── docs/
│   ├── original-docs/          # Source papers and research materials
│   ├── analysis/               # Research analysis and notes
│   └── summaries/              # Key findings and summaries
├── src/                        # Implementation code
├── data/                       # Datasets and experimental data
├── results/                    # Experimental results and outputs
├── notebooks/                  # Jupyter notebooks for analysis
├── scripts/                    # Utility and automation scripts
└── CLAUDE.md                   # This file
```

## Key Research Areas

### Tool-Integrated Reasoning (TIR)
- Code execution integration with LLM reasoning
- Multi-turn RL training with tool feedback
- Verifiable vs non-verifiable domain handling

### Reinforcement Learning Approaches
- DAPO (improved GRPO) for judge training
- Reward design combining correctness, format, and tool-specific rewards
- Iterative training strategies

### Evaluation Methodologies
- Pointwise evaluation (scoring individual responses)
- Pairwise evaluation (comparing response pairs)
- Listwise evaluation (ranking multiple responses)

## Implementation Guidelines

### Development Environment
- Python 3.8+
- PyTorch for deep learning components
- Code execution sandbox for tool integration
- Evaluation frameworks for judge benchmarks

### Coding Standards
- Follow Google's Python style guide
- Comprehensive docstrings for all functions
- Type hints for better code maintainability
- Unit tests for core functionality

### Experimental Tracking
- Use wandb or similar for experiment logging
- Version control for data and models
- Reproducible experimental setups

## Research Questions

1. How effective is tool integration compared to text-only reasoning?
2. What is the optimal balance between different reward components?
3. Can self-bootstrapping match distillation performance?
4. How does performance scale with model size and training data?

## Next Steps

1. **Literature Review**: Complete analysis of related work in tool-augmented LLMs
2. **Baseline Implementation**: Implement core TIR-Judge framework
3. **Experimental Design**: Set up evaluation pipelines for multiple benchmarks
4. **Ablation Studies**: Systematic analysis of key components

## Notes for Claude

- Focus on reproducible research practices
- Prioritize clear documentation and analysis
- Consider computational constraints and efficiency
- Maintain research integrity and proper attribution
- Use systematic approach for experimental validation

## Resources

- Original paper: `./docs/original-docs/2510.23038v1.pdf`
- Converted markdown: `./docs/original-docs/2510.23038v1.md`
- Research notes: `./docs/original-docs/`

---

*This document serves as a guide for understanding and extending the TIR-Judge research. Update as the project evolves.*