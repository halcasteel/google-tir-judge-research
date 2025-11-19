<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Google's TIR-Judge paper search and find

The Google TIR-Judge paper is officially titled "Incentivizing Agentic Reasoning in LLM Judges via Tool-Integrated Reinforcement Learning." It was released on arXiv in October 2025 and describes a novel framework for training large language model (LLM) judges to integrate code execution with text-based reasoning for more verifiable and accurate evaluations.[^1_1][^1_2][^1_3]

### Paper Details and Abstract

TIR-Judge is built on principles including diverse training across verifiable and non-verifiable domains, flexible judgment formats (pointwise, pairwise, listwise), and reinforcement learning that allows the judge to evolve without distilled trajectories. The framework leverages a code executor to validate factual or computational claims and demonstrates strong results: TIR-Judge outperforms existing reasoning-based reward models in several benchmarks by significant margins, while being highly parameter efficient (matching Claude-Opus-4 performance with an 8B parameter model in certain tasks).[^1_2][^1_3][^1_1]

On seven public benchmarks, TIR-Judge surpasses strong reasoning-based judges by up to 6.4% (pointwise) and 7.7% (pairwise), achieving listwise performance comparable to much larger models. The work proves that reinforced tool-use can allow LLM judges to "self-evolve" without the need for label distillation or high-cost ultra-large models.[^1_3][^1_1][^1_2]

### Technical Blueprint References

- arXiv abstract and full paper: [arxiv.org/abs/2510.23038](https://arxiv.org/abs/2510.23038)
- Direct PDF: [arxiv.org/pdf/2510.23038.pdf](https://www.arxiv.org/pdf/2510.23038.pdf)
- Project summary and video overview: [YouTube: TIR-Judge - RL for LLM Judges with Code Execution](https://www.youtube.com/watch?v=0g5wQnQ6KQQ)[^1_4]
- Social/technical discussion: [Rohan Paul on X](https://x.com/rohanpaul_ai/status/1983819664716083462), [LinkedIn post](https://www.linkedin.com/posts/anthony-alcaraz-b80763155_a-new-paper-from-google-just-solved-the-llm-activity-739501836365737574%E2%80%A6)[^1_5][^1_6]


### Stronger Prompt for Next Steps

To drive a follow-up technical deep dive into TIR-Judge for agentic multi-agent system design, consider this prompt:

- "Extract the full technical blueprint and design paradigm of TIR-Judge, including training methodology, reinforcement learning structures, integration with code executors, judgement format flexibility, and comparative evaluation metrics from the official arXiv paper and supplementary resources. Reference supporting papers and provide mapped lessons for agentic multi-agent architectures."

This will enable a systematic, reference-backed walkthrough suitable for your technical outlining and design mapping workflows.

If you want direct content extractions (tables, algorithms, evaluation metrics), request:

- "Show the main experiment results, architecture diagrams, algorithms, and significant tables from the TIR-Judge arXiv paper with full reference mapping for technical blueprint analysis."
<span style="display:none">[^1_10][^1_11][^1_12][^1_13][^1_14][^1_15][^1_16][^1_17][^1_18][^1_19][^1_20][^1_7][^1_8][^1_9]</span>

<div align="center">⁂</div>

[^1_1]: https://arxiv.org/abs/2510.23038

[^1_2]: https://arxiv.org/html/2510.23038v1

[^1_3]: https://www.arxiv.org/pdf/2510.23038.pdf

[^1_4]: https://www.youtube.com/watch?v=0g5wQnQ6KQQ

[^1_5]: https://x.com/rohanpaul_ai/status/1983819664716083462

[^1_6]: https://www.linkedin.com/posts/anthony-alcaraz-b80763155_a-new-paper-from-google-just-solved-the-llm-activity-7395018363657375744-S48y

[^1_7]: https://arxiv.org/html/2511.01833v1

[^1_8]: https://arxiv.org/html/2509.23292v2

[^1_9]: https://patents.google.com/patent/US10332315B2/en

[^1_10]: https://arxiv.org/html/2510.05592

[^1_11]: https://patents.google.com/patent/CA3173966A1/en

[^1_12]: https://arxiv.org/pdf/2510.26167.pdf

[^1_13]: https://patents.google.com/patent/WO2011106798A1/en

[^1_14]: https://arxiv.org/html/2509.14257v2

[^1_15]: https://patents.google.com/patent/US20190188471A1/en

[^1_16]: https://www.facebook.com/datasciencedojo/posts/-the-way-we-evaluate-llms-is-fundamentally-broken-and-this-paper-makes-a-compell/856177960265690/

[^1_17]: https://arxiv.org/pdf/2508.13167.pdf

[^1_18]: https://patents.google.com/patent/US7170670B2/en

[^1_19]: https://x.com/ritaranx/status/1986218902368755897

[^1_20]: https://www.arxiv.org/pdf/2509.14257v1.pdf

