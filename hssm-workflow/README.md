# hssm-workflow

An [Agent Skill](https://agentskills.io) for choice/reaction-time inference with
HSSM, using the same workflow, references and shared diagnostic-reporting
structure as Baygent's existing skills.

The initial implementation targets HSSM 0.5.0 and a flat analytical DDM with
complete positive RTs in seconds, responses `-1/+1`, and no lapse mixture.
It includes data/parameter guidance, native HSSM API recipes and a small adapter
for separate RT and choice marginal predictive assessments.
Numerical runs and agent evaluations are deferred; this is not a claim of
validated parameter recovery or a tested dependency stack.

## Install

Copy **both `hssm-workflow` and `bayesian-workflow`** into the skill directory
used by your agent. The Bambi skill is optional. For example:

```bash
git clone https://github.com/Learning-Bayesian-Statistics/baygent-skills.git /tmp/baygent-skills
mkdir -p ~/.claude/skills
cp -r /tmp/baygent-skills/hssm-workflow ~/.claude/skills/
cp -r /tmp/baygent-skills/bayesian-workflow ~/.claude/skills/
```

For another compatible agent or a project-local installation, use its configured
skill directory. The workflow resolves the dependency location and works from
the installed skill folders; repository notebooks and evaluation files are
optional development resources. Copying a skill does not install HSSM or its
Python dependencies.

## Example prompts

- “Inspect my choice and reaction-time data for a flat analytical DDM with
  HSSM. Help me choose priors and a non-decision-time bound before sampling.”
- “I have an HSSM DDM fit. Check response proportions and conditional RT
  distributions, then report diagnostic limitations.”
- “Write an HSSM analysis for this real dataset, but leave sampling and
  predictive generation for later.”

## What's included

```text
hssm-workflow/
├── SKILL.md
├── README.md
├── scripts/
│   └── prepare_rt_choice.py
└── references/
    ├── data-and-ddm.md
    ├── priors-and-links.md
    └── prediction-and-reporting.md
```

See [SKILL.md](SKILL.md) for scope and the installed helper/reporting handoff.
Hierarchical models, LANs, RLSSMs, other SSM families and special response
regimes require subsequent scoped verification.

## License

These skill instructions are MIT, as distributed with baygent-skills. HSSM's
own software license is separate; this skill does not relicense that package.
