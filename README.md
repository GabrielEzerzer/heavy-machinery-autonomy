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

- **Cutting.** Two sites mark the ends of the bucket's cutting edge, taken from the bucket mesh. Between two physics steps the edge sweeps a four-sided patch, and every cell whose centre lies inside is lowered to the edge's height there, so a fast edge cannot skip cells. An edge sliding along its own length or moving straight down sweeps no area and removes nothing. An edge end that moves faster than 10 m/s between two physics steps, as after a reset or a pose set by hand, cuts nothing that step and starts again from where it is. A bucket placed below ground before a reset cuts nothing until it moves, but the soil holds it from the start.
- **Bucket.** Soil enters the bucket only while the edge moves towards the bucket's opening, up to the 0.45 m³ heaped capacity of the standard bucket ([Hitachi ZX120 specification sheet KA-EN200ID](https://machineryline.kr/img/pdf/7/3/1688885834710464825/KA-EN200ID_ZX120.pdf)). Soil cut by the back of the bucket, or when it is full, is left ahead of the edge as loose spoil.
- **Dumping.** With the edge out of the ground, the bucket pours once its lip points more than 45° below horizontal. The rate grows with the tilt past that angle, up to a full bucket in 0.5 s with the lip straight down. The lip passes 45° at a pitch of about 1.48 rad in `inverse_kinematics`, when the opening faces about 5° below horizontal, and points straight down at about 0.75 rad, opening 47° down. The demo's dump pose (0.9 rad) empties a full bucket in about 0.7 s. Swinging through the dump angle for one physics step sheds about 0.25% of a full bucket. A bucket turned upside down with its edge still in the ground keeps its load: with the edge 5 cm deep and the lip straight down, a full bucket holds all 0.45 m³ for 2 s.
- **Where soil lands.** Spills and dumps spread over the area under the edge, at least 0.3 m in each direction. They never land on the ground under the tracks: that share goes to the rest of the same area, or to the nearest cells outside the tracks if the whole area is under them.
- **Settling.** Only loose spoil slides, and only towards neighbours steeper than the soil's friction angle. Cut walls stand, and spoil does not slide under the tracks.
- **Volume.** Ground, bucket and `lost_volume` always add up to the starting volume. `lost_volume` counts soil that leaves the grid and soil that would pile above the ground's top, 2 m above the starting surface.
- **Price of a cut.** The resistance is the fundamental earthmoving equation as written in [arXiv:2505.19330](https://arxiv.org/abs/2505.19330), Section 1.2, plus the tooth penetration term of [Egli et al. 2022](https://doi.org/10.1109/LRA.2022.3189834). Its inputs are the cut depth along the edge, the rake of the bucket's lip against the horizontal, and the soil in the bucket as surcharge. The failure angle is the one that minimises the weight factor N<sub>γ</sub> at the current rake (same section, Eq. 16), but never less than the passive angle π/4 − φ/2, read from a table built once per soil. The equation is singular as the rake nears π − φ − δ − β (same paper); past a rake limit 0.1 rad short of that, the factors stay at their value at the limit. The force is capped smoothly at 175 kN.
- **Spoil.** Cohesion, adhesion and penetration act only on the part of the cut below the loose spoil. Spoil adds its weight and surcharge but no cohesion, so recutting a heap costs far less than cutting the bank: in the stiffest soil a 0.25 m drag meets 134 kN in undisturbed ground and about 4 kN in spoil. The split is taken at each point along the edge, so the force changes smoothly where spoil meets undisturbed ground.
- **How the force acts.** Three MuJoCo friction-loss tendons run from the bucket tip along axes set by the bucket's pose alone: h, horizontal and across the edge; s, horizontal and along the edge; v, vertical. Before every physics step each axis gets its own limit from the pose and the ground, never from the motion or the command:
  - h: the full separation term plus the penetration term times the horizontal part of the lip, for the deeper of the ground one cell ahead of, one cell behind and at each point of the edge.
  - s: the same, for the ground one cell beyond each end of the edge. A sideways sweep into undisturbed ground costs at least as much as cutting forward at that depth.
  - v: the full separation term plus the penetration term times the vertical part of the lip, for the deepest ground around the edge, ends included.

  All three are zero when no ground around the edge is above it. Friction only resists, so the soil never pushes, the force applies at any speed, and stiff soil stalls the arm against its joint force limits. The machine model sets MuJoCo's solver tolerance to 1e-10; at the default the solver can stop after one iteration where the edge starts or stops slipping.
- **Against the published forces.** A 0.3 m drag through the nominal soil meets about 58 kN on the h axis, and the demo's drag about 51 kN. Both are below the arm crowd force of 63 kN (SAE) to 65 kN (ISO), and the bucket digging force of 86 kN (SAE) to 99 kN (ISO), on the specification sheet. The stiffest soil, at about 149 kN, stalls the arm.
- **Weight.** Soil in the bucket weighs on it as a downward force at the bucket's centre of mass.
- **Soil parameters.** `hma.soil.sample_soil(seed)` draws the six soil parameters of Egli et al. 2022, Table I; `NOMINAL` is the middle of each range.
- **State.** A `Soil` follows one MuJoCo `MjData` at a time. To reset, reset the simulation state first, then call `soil.reset(model, data)`. With a released base, the protected ground under the tracks and the collision heights follow the machine once per control period, not every physics step.

Limits:

- Quasi-static. The force has no rate term, and soil in the bucket adds weight but no mass.
- 2.5D. There is one height per cell, so no overhangs or tunnels.
- No flow inside the bucket. Its contents are a single volume.
- No wall collapse. Cut walls stand at any slope; real trench walls can fail.
- Only the cutting edge meets the soil. The back and sides of the bucket and the arm pass through the ground without force.
- Analytic digging models of this kind match measured forces to about 20% at best ([Egli, Terenzi and Hutter 2024](https://www.research-collection.ethz.ch/server/api/core/bitstreams/c8a1f541-96da-47ac-82bd-950de01a40ad/content), citing Cannon and Singh 1999 and Luengo et al. 1998). Nothing here is fitted to measurements.
- The adhesion factor N<sub>ca</sub> is kept at zero or above, which the source does not do. Its negative values at small rakes would make the force negative, so the edge would cut for nothing. For a 0.3 m cut that happens at some rake for 69 of 1000 sampled soils with the full penetration term, and for 334 without it; at some depth between 2 cm and 0.5 m, for 528. The floor acts below a rake of 0.37 rad in the nominal soil, the rake the demo cuts at, and below up to 0.63 rad in the least frictional soils.
- These values are assumptions, not taken from a source: the 2 cm edge thickness in the penetration term, the 45° dump angle, the 0.5 s pour time, the 0.3 m smallest spill area, the 175 kN force cap, treating loose spoil as cohesionless, and a repose angle equal to the friction angle. The force and the cut both use the 0.61 m edge of the bucket mesh, against 0.85 m for the bucket on the specification sheet, so the standard bucket would cut and resist about 40% more per pass.
- MuJoCo friction is a soft constraint, so a held edge still creeps: at about 0.23 mm/s resting in soil, and about 4 mm/s against a stall in stiff soil, where the arm is at its force limit and the tip force, 88 kN, is about 66% of the nearest axis limit.
- An edge commanded slower than about 3 mm/s never loads its joints fully. At that scale each joint's speed control is soft too: at about 1 mm/s into stiff soil the arm pushes at about 28% of its force limit, and the soil gives way at that load.
- The force is set by the pose, not the motion, which costs fidelity in four ways. The numbers are for the nominal soil unless stated.
  - Each axis resists equally both ways, so backing away from a cut face costs about as much as cutting forward. From a face priced at 64 kN, backing away 5, 10 and 20 cm takes 1.05, 0.84 and 0.66 times the work of cutting forward that far, at up to 68 kN.
  - Moving the back of the bucket first is priced as if the lip led. There is no back-of-blade rake.
  - Diagonal motion pays each axis it moves along, so the total force can reach the root of the sum of the squared limits. Over the demo cycle it reaches 1.41 times the h limit, and 2.39 times over 30 random soils.
  - A bucket at rest in soil is held up to the limits on all axes, and lifting out pays the v limit. Lifting out of a 0.5 m slot takes 2.1 s and 12 kJ at up to 69 kN, 0.30 times the work of a 0.5 m forward cut; in the stiffest soil the boom lifts it 5 cm in 15 s. In 15 of 30 random soils at least one phase of the demo cycle (curl, lift, swing or drag) runs to its 15 s timeout, though all 30 complete a dump.
- A cell is lowered when the edge passes its centre, so the first motion from a pose on a row of cell centres can cut up to 5 litres for almost no work.
- Ground under the tracks is never cut. That does not fix [issue 6](../../issues/6): a trench edge just outside the tracks still gives too low a tipping load on a free base.

## Limits

- Simulation only. Nothing here has been checked against a real excavator.
- The soil is a height grid and a force formula, not grains.
- Calibrating the controller to a specific real machine is future work.

## Licence

The code is under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): you may read, run and change it for noncommercial purposes. Third-party files keep their own licences and are credited where they are added. The machine model is converted from [pwri-opera/zx120_ros](https://github.com/pwri-opera/zx120_ros) under Apache-2.0; its [notice](src/hma/machine/assets/NOTICE) lists the changes.

Required Notice: Copyright 2026 Gabriel Ezerzer
