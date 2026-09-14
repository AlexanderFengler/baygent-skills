# baygent-skills

A set of skills to call your agent Bayes. Thomas Bayes.

[Agent Skills](https://agentskills.io) for Bayesian modeling, causal inference, and probabilistic thinking. Compatible with Claude Code, Kimi Code, Cursor, Gemini CLI, and any agent that supports the [Agent Skills spec](https://agentskills.io/specification).

## Available skills

| Skill | Description |
|---|---|
| [bayesian-workflow](bayesian-workflow/) | Full Bayesian modeling workflow with PyMC and ArviZ. [Full breakdown](https://learnbayesstats.com/blog-posts/bayesian-workflow-agent-skill-pymc-arviz). |
| [bambi-workflow](bambi-workflow/) | Formula-based Gaussian and hierarchical Bernoulli regression with Bambi; shared Bayesian diagnostics and reporting. [Two interactive examples](examples/bambi-workflow/). |
| [hssm-workflow](hssm-workflow/) | Flat analytical DDM with HSSM, choice/RT checks and shared reporting. [Interactive example](examples/hssm-workflow/) with [runtime evidence](evals/hssm-workflow/iteration-1/runtime/README.md); agent behavior evaluation pending. |
| [causal-inference](causal-inference/) | Production-grade Bayesian causal inference with PyMC, CausalPy, and DoWhy. [Full breakdown](https://learnbayesstats.com/blog-posts/causal-inference-agent-skill-pymc-causalpy-dowhy). |
| [amortized-workflow](amortized-workflow/) | Amortized Bayesian workflow with BayesFlow for simulation-based inference. [Full breakdown](https://learnbayesstats.com/blog-posts/amortized-bayesian-inference-agent-skill-bayesflow). |

More skills coming soon. Issues and PRs are welcome!

## Quick install

### Claude Code

```bash
git clone https://github.com/Learning-Bayesian-Statistics/baygent-skills.git /tmp/baygent-skills
mkdir -p ~/.claude/skills
cp -r /tmp/baygent-skills/bayesian-workflow ~/.claude/skills/
cp -r /tmp/baygent-skills/bambi-workflow ~/.claude/skills/  # requires bayesian-workflow
cp -r /tmp/baygent-skills/hssm-workflow ~/.claude/skills/  # requires bayesian-workflow
cp -r /tmp/baygent-skills/causal-inference ~/.claude/skills/
cp -r /tmp/baygent-skills/amortized-workflow ~/.claude/skills/  # BayesFlow / SBI
```

### Other compatible agents

Clone the repo and copy the skill folders you need into your agent's skills location:

```bash
git clone https://github.com/Learning-Bayesian-Statistics/baygent-skills.git /tmp/baygent-skills
cp -r /tmp/baygent-skills/bayesian-workflow/ ~/.config/agents/skills/bayesian-workflow/
cp -r /tmp/baygent-skills/bambi-workflow/ ~/.config/agents/skills/bambi-workflow/
cp -r /tmp/baygent-skills/hssm-workflow/ ~/.config/agents/skills/hssm-workflow/
cp -r /tmp/baygent-skills/causal-inference/ ~/.config/agents/skills/causal-inference/
cp -r /tmp/baygent-skills/amortized-workflow/ ~/.config/agents/skills/amortized-workflow/
```

> **Note:** bambi-workflow, hssm-workflow and causal-inference each depend directly on bayesian-workflow — install the shared skill alongside each. Bambi examples target Bambi 0.21 / PyMC 6; see their [setup and run instructions](examples/bambi-workflow/README.md). HSSM has a separate [tested runtime stack and example](examples/hssm-workflow/README.md); its skill does not require the Bambi skill.

## Philosophy

These skills are **opinionated and workflow-first**. They don't just teach an agent what PyMC functions exist — they enforce a specific sequence of steps (prior predictive checks, diagnostics, calibration, reporting) and guardrails (94% HDI, reproducible seeds, save-to-disk) that produce reliable analyses.

Each skill is focused and lean. Rather than one monolithic skill that covers everything, we build specialized skills that do one thing well:

- **bayesian-workflow** covers the fundamentals that every Bayesian analysis needs.
- **bambi-workflow** owns Bambi formulas, priors and native predictions, and delegates shared checks and reports to bayesian-workflow.
- **hssm-workflow** owns HSSM likelihoods, parameter priors and choice/RT predictions, and reuses the shared Bayesian diagnostics and report structure.
- **causal-inference** handles causal design, identification, and refutation — delegating the modeling to bayesian-workflow.
- **amortized-workflow** covers simulation-based inference with BayesFlow — end-to-end architecture selection, training, simulation-based diagnostics, and real data application.

## About

Created by [Alexandre Andorra](https://alexandorra.github.io/), host of [Learning Bayesian Statistics](https://www.learnbayesstats.com/).

## License

MIT - see [LICENSE](LICENSE).
