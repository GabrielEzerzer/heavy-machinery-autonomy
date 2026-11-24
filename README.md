# Heavy Machinery Autonomy

A simulation demo. You type a digging task in plain English, and a simulated 12-tonne excavator digs it. One controller, trained in simulation, drives every individual machine of this type with no retuning.

**Status: in development.** The machine model, the soil and the language step run on their own; a scripted dig cycle runs, but no controller digs yet. The [feature map](#feature-map) lists what is built and what is planned.

## How it works

1. A GPT model turns an instruction such as "dig 1 m down at the far marker" into a small task specification: which marker, how deep, how long.
2. Fixed checks accept or refuse the specification: valid fields, within the machine's reach, within trench safety rules. A person confirms before anything moves.
3. A dig planner turns the specification into paths for the bucket.
4. A learned controller follows those paths. Every run draws a different machine, with its own hydraulic delay, dead zone, flow limit, slack in the joints, weights and soil. The controller is the same for all of them.

The physics is [MuJoCo](https://mujoco.org). The soil is a grid of heights with an analytic digging force. The controller is trained with PPO from [Stable-Baselines3](https://stable-baselines3.readthedocs.io).

## Feature map

A feature is **done** when it runs from a clean checkout and its checks pass.

| Feature | What it does | Status |
|---|---|---|
| Machine model | A 12 t class excavator with swing, boom, arm and bucket joints. Each joint takes a speed command and has a force limit. The base can be held still or left free to slide and tip | Done |
| Hydraulics model | Gives each machine its own delay, sluggishness, dead zone, lever-to-speed curve, flow limit, force limit and slack in the joints | Planned |
| Soil | A grid of ground heights that the bucket cuts, fills from and dumps onto, with a digging force that depends on the soil | Done |
| Dig planner | Turns a trench into repeated cycles: cut, drag, curl, lift, swing, dump | Planned |
| Baseline controller | A conventional controller tuned once on the standard machine | Planned |
| Learned controller | One network that follows the planner's paths on any machine drawn | Planned |
| Training | One command that trains the controller in stages of rising difficulty | Planned |
| Evaluation | Scores a controller on machines it never saw in training | Planned |
| Language to task | A GPT model fills a task specification from a plain-English instruction | Done |
| Plan checks | Refuses a specification that is malformed, out of reach or against trench safety rules, and asks a person to confirm | Done, with a simplified arm for the reach check until the machine model is connected |
| Demo screen | The plan, the drawn machine's card, both controllers side by side and the results chart | Planned |
| Balance on rough ground | The machine can slide and tip, and the controller learns to dig without unsettling it | Planned, after the first working demo |

Not planned: other makes, sizes or arm geometries; soil simulated as grains; buried obstacles.

## Try it

Python 3.12 or newer.

```
pip install -e .
python -m hma.machine.demo
python -m hma.soil.demo
python -m hma.task "dig 1 m down at the far marker"
```

The first command opens a viewer and moves the arm through its range. The second runs one scripted dig cycle (lower, drag, curl, lift, swing, dump) in nominal soil. The third asks a GPT model for a plan and checks it. It calls the [Codex CLI](https://github.com/openai/codex), which must be installed and signed in with your own ChatGPT account; this repo holds no API key.

## Soil

Each machine has its own ground: the model's height field, with 10 cm cells, rewritten once per control step.

- **Cutting.** Two sites mark the ends of the bucket's cutting edge, taken from the bucket mesh. Between two physics steps the edge sweeps a four-sided patch, and every cell whose centre lies inside is lowered to the edge's height there, so a fast edge cannot skip cells. An edge sliding along its own length or moving straight down sweeps no area and removes nothing.
- **Bucket.** Soil enters the bucket only while the edge moves towards the bucket's opening, up to the 0.45 m³ heaped capacity of the standard bucket ([Hitachi ZX120 specification sheet KA-EN200ID](https://machineryline.kr/img/pdf/7/3/1688885834710464825/KA-EN200ID_ZX120.pdf)). Soil cut by the back of the bucket, or when it is full, is left ahead of the edge as loose spoil. Tipping the lip more than 45° below horizontal with the edge out of the ground empties the bucket onto the ground beneath.
- **Settling.** Only loose spoil slides, and only towards neighbours steeper than the soil's friction angle. Cut walls stand.
- **Volume.** Ground, bucket and anything that leaves the grid always add up to the starting volume.
- **Force.** The resistance is the fundamental earthmoving equation as written in [arXiv:2505.19330](https://arxiv.org/abs/2505.19330), plus the tooth penetration term of [Egli et al. 2022](https://doi.org/10.1109/LRA.2022.3189834). Its inputs are the cut depth along the edge, the rake of the bucket against its travel, and the soil in the bucket as surcharge. The equation is singular as the rake nears π − φ − δ − β (same paper); there the force ramps smoothly to twice its last regular value instead.
- **How the force acts.** It works as MuJoCo friction loss on three tendons from the bucket tip, along axes aligned with the edge's motion. Each is capped at 173 kN, so the total never exceeds 300 kN. Friction only resists, so stiff soil stalls the arm against its joint force limits. The force fades to zero as the commanded edge speed goes to zero, so an idle bucket is held only by its own joints.
- **Weight.** Soil in the bucket weighs on it as a downward force at the bucket's centre of mass.
- **Soil parameters.** `hma.soil.sample_soil(seed)` draws the six soil parameters of Egli et al. 2022, Table I; `NOMINAL` is the middle of each range.

Limits:

- Quasi-static. The force has no rate term, and soil in the bucket adds weight but no mass.
- 2.5D. There is one height per cell, so no overhangs or tunnels.
- No flow inside the bucket. Its contents are a single volume.
- No wall collapse. Cut walls stand at any slope; real trench walls can fail.
- Only the cutting edge meets the soil. The back and sides of the bucket and the arm pass through the ground without force.
- Analytic digging models of this kind match measured forces to about 20% at best ([Egli, Terenzi and Hutter 2024](https://www.research-collection.ethz.ch/server/api/core/bitstreams/c8a1f541-96da-47ac-82bd-950de01a40ad/content), citing Cannon and Singh 1999 and Luengo et al. 1998). Nothing here is fitted to measurements.
- These values are assumptions, not taken from a source: the 2 cm edge thickness in the penetration term, the 45° dump angle, the force cap, and a repose angle equal to the friction angle. The edge is 0.61 m wide in the mesh, against 0.85 m for the bucket on the specification sheet.
- Ground under the tracks is never cut. That does not fix [issue 6](../../issues/6): a trench edge just outside the tracks still gives too low a tipping load on a free base.

## Limits

- Simulation only. Nothing here has been checked against a real excavator.
- The soil is a height grid and a force formula, not grains.
- Calibrating the controller to a specific real machine is future work.

## Licence

The code is under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): you may read, run and change it for noncommercial purposes. Third-party files keep their own licences and are credited where they are added. The machine model is converted from [pwri-opera/zx120_ros](https://github.com/pwri-opera/zx120_ros) under Apache-2.0; its [notice](src/hma/machine/assets/NOTICE) lists the changes.

Required Notice: Copyright 2026 Gabriel Ezerzer
