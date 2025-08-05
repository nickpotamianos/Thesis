---
applyTo: '**'
---
Below is a concise **instruction sheet** you can paste as the *system* (or *pre-prompt*) for your VS Code Agent.
It tells the agent **exactly what to do and—equally important—what *not* to do** while implementing the SotA pipeline you sketched.
Feel free to tweak paths or wording, but resist adding new requirements unless you really need them.

```
# ───────────────────────────────────────────────────────────────
#  VS CODE AGENT – EXECUTION GUIDELINES
#  Context: implementing the “sota/” pipeline for the MILUV thesis
# ───────────────────────────────────────────────────────────────

## 0. General behaviour
1. **Obey the user’s explicit file-creation instructions verbatim.**  
   - Do **not** invent extra files (e.g., README.md, demos, simulations, notebooks) unless the user later *explicitly* asks for them.  
   - Do **not** rename, move, or delete anything beyond what is stated.

2. **Keep the original “miluv/” dev-kit untouched** (read-only), except where the user later requests edits.

3. **No self-initiated scaffolding** (e.g., “Let’s add CI”, “Generate examples”, “Run a test”).  
   Execute only the concrete steps listed below—or future steps that the user spells out.

4. **All new code must follow the exact directory tree given.**  
   Reproduce folders/files exactly; do not add placeholders.

5. **No automatic documentation generation.**  
   You may add doc-strings and inline comments inside the code files you create, but do not spawn separate .md, .rst, or .txt docs.

## 1. Repository re-organisation (create only what is asked)
Create the following top-level folder **sota/**  
(exists alongside the original “miluv/”):

```

sota/
├── README.md                 # ← create only because the user listed one
├── env/
│   └── environment\_sota.yml
├── data\_prep/
│   ├── uwb\_extract\_dataset.py
│   ├── imu\_extract\_dataset.py
│   └── vision\_extract\_dataset.py           # ← leave empty for now
├── uwb/
│   ├── models/
│   │   ├── cnn\_nlos.py
│   │   └── cnn\_bias\_regressor.py
│   ├── train\_nlos\_classifier.py
│   ├── train\_bias\_regressor.py
│   └── infer\_uwb.py
├── imu/
│   ├── repiln/
│   │   ├── model.py
│   │   ├── train\_repiln.py
│   │   └── infer\_repiln.py
│   └── utils.py
├── factor\_graph/
│   ├── factors.py
│   ├── fg\_runner.py
│   └── config.yaml            # ← create empty stub
└── scripts/
├── run\_training\_all.sh
└── run\_online\_demo.sh

```

*Do not create any other directories* (e.g., “tests”, “docs”) unless later instructed.

## 2. Populate files with the exact code blocks supplied
– Copy each Python / shell snippet from the user’s message **as-is** into its respective file.  
– Maintain original indentation and imports.  
– Do not “improve”, “refactor”, or “auto-format” beyond basic PEP8 spacing.

## 3. environment_sota.yml
Paste the YAML exactly as written (same package versions).  
Don’t add/remove dependencies.

## 4. Shebangs & execution bits
For every `*.sh` created, prepend `#!/usr/bin/env bash` and make it executable.  
No other shell scripts.

## 5. Version control
Commit only the files above.  
No `.vscode/`, `.idea/`, virtual-env folders, cache files, or data outputs.

## 6. Running code
The agent is **not** to run any script or training. Implementation only.

## 7. When uncertain
If a future user instruction is ambiguous:
   – Pause and ask for clarification *instead of guessing*.

# ───────────────────────────────────────────────────────────────
#  End of VS Code Agent instructions
# ───────────────────────────────────────────────────────────────
```