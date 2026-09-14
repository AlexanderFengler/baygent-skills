# bambi-workflow

An [Agent Skill](https://agentskills.io) for formula-based Bayesian regression
with Bambi, from prior checks through response-scale interpretation and a
canonical diagnostic report.

The initial scope is Gaussian regression and hierarchical Bernoulli regression
with prediction for an observed group. It targets Bambi 0.21.0, PyMC 6,
ArviZ 1.x, and Python 3.12. Broader families, distributional regression,
splines/HSGP, and unseen-group prediction are later extensions.

## Install

Install **both `bambi-workflow` and `bayesian-workflow`**. The latter supplies
diagnostic utilities and the report template. Copy the folders from
[baygent-skills](https://github.com/Learning-Bayesian-Statistics/baygent-skills)
into the skill directory used by your agent, for example:

```bash
git clone https://github.com/Learning-Bayesian-Statistics/baygent-skills.git /tmp/baygent-skills
mkdir -p ~/.claude/skills
cp -r /tmp/baygent-skills/bambi-workflow ~/.claude/skills/
cp -r /tmp/baygent-skills/bayesian-workflow ~/.claude/skills/
```

For another compatible agent or a project-local installation, use that agent's
configured skill directory. The workflow resolves the dependency location and
does not require the source checkout, repository examples, or evaluation files.
Install the supported Python packages in an analysis environment; copying skill
folders does not install Bambi or its dependencies.

## Example prompts

- “Use Bambi for a Gaussian regression and compare the expected outcome at
  two predictor values. Check the priors and report diagnostic limitations.”
- “Fit a hierarchical logistic regression of survey support by age and income,
  pooling across regions. Use low income as the reference and predict support
  for an urban respondent aged 40 in an observed region.”
- “Inspect my Bambi model's resolved priors and help me interpret a
  probability-scale prediction with uncertainty.”

## What's included

```text
bambi-workflow/
├── SKILL.md
├── README.md
└── references/
    ├── formula-syntax.md
    ├── families-and-links.md
    ├── priors-in-bambi.md
    └── interpretation-and-reporting.md
```

The skill uses native Bambi functions and the shared Bayesian diagnostic
scripts. It adds no model-serialization layer or duplicate diagnostic harness.
See [SKILL.md](SKILL.md) for scope, dependency resolution, and the workflow.

## License

MIT, as distributed with baygent-skills.

