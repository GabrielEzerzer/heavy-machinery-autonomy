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

- **Cutting.** Two sites mark the ends of the bucket's cutting edge, taken from the bucket mesh. Between two physics steps the edge sweeps a four-sided patch, and every cell whose centre lies inside is lowered to the edge's height there, so a fast edge cannot skip cells. An edge sliding along its own length or moving straight down sweeps no area and removes nothing. An edge end that moves faster than 10 m/s between two physics steps, as after a reset or a pose set by hand, cuts nothing that step and starts again from where it is. A bucket placed below ground before a reset cuts nothing until it moves; from its first motion it cuts and meets the full force.
- **Bucket.** Soil enters the bucket only while the edge moves towards the bucket's opening, up to the 0.45 m³ heaped capacity of the standard bucket ([Hitachi ZX120 specification sheet KA-EN200ID](https://machineryline.kr/img/pdf/7/3/1688885834710464825/KA-EN200ID_ZX120.pdf)). Soil cut by the back of the bucket, or when it is full, is left ahead of the edge as loose spoil.
- **Dumping.** With the edge out of the ground, the bucket pours once its lip points more than 45° below horizontal. The rate grows with the tilt past that angle, up to a full bucket in 0.5 s with the lip straight down. The lip passes 45° at a pitch of about 1.45 rad in `inverse_kinematics`, when the opening faces about 7° below horizontal, and points straight down at about 0.75 rad, opening 47° down. The demo's dump pose (0.9 rad) empties a full bucket in about 0.7 s. Swinging through the dump angle for one physics step sheds about 0.25% of a full bucket.
- **Where soil lands.** Spills and dumps spread over the area under the edge, at least 0.3 m in each direction. They never land on the ground under the tracks: that share goes to the rest of the same area, or to the nearest cells outside the tracks if the whole area is under them.
- **Settling.** Only loose spoil slides, and only towards neighbours steeper than the soil's friction angle. Cut walls stand, and spoil does not slide under the tracks.
- **Volume.** Ground, bucket and `lost_volume` always add up to the starting volume. `lost_volume` counts soil that leaves the grid and soil that would pile above the ground's top, 2 m above the starting surface.
- **Force.** The resistance is the fundamental earthmoving equation as written in [arXiv:2505.19330](https://arxiv.org/abs/2505.19330), Section 1.2, plus the tooth penetration term of [Egli et al. 2022](https://doi.org/10.1109/LRA.2022.3189834). Its inputs are the cut depth along the edge, the rake of the bucket against its travel, and the soil in the bucket as surcharge. The depth is read one cell ahead of the edge, or at the edge itself where that point is off the grid. The failure angle is the one that minimises the weight factor N<sub>γ</sub> at the current rake (same section, Eq. 16), read from a table built once per soil. The equation is singular as the rake nears π − φ − δ − β (same paper); there the force ramps smoothly to twice its last regular value instead. The force is capped smoothly at 175 kN.
- **Spoil.** Cohesion, adhesion and penetration act only on the part of the cut below the loose spoil. Spoil adds its weight and surcharge but no cohesion, so recutting a heap costs far less than cutting the bank: in the stiffest soil a 0.25 m drag meets 137 kN in undisturbed ground and about 4 kN in spoil. The split is taken at each point along the edge, so the force changes smoothly where spoil meets undisturbed ground.
- **Back of the bucket.** The back-of-blade rake applies only while the edge moves more than about 120° away from the way the bucket's opening faces, blended over about 15° of motion direction. Lowering the bucket teeth first, and sweeping it sideways, meet the cutting-rake force: a sideways sweep costs the same as cutting forward at that depth.
- **Against the published forces.** A 0.3 m drag through the nominal soil meets about 52 kN at the edge. That is below the arm crowd force of 63 kN (SAE) to 65 kN (ISO), and the bucket digging force of 86 kN (SAE) to 99 kN (ISO), on the specification sheet. The stiffest soil, at about 153 kN, stalls the arm.
- **How the force acts.** It works as MuJoCo friction loss on three perpendicular tendons from the bucket tip, each capped at the force divided by √3. The three axes form a cube whose diagonal points along the edge's motion. A sliding edge therefore meets exactly the force, straight against its motion, and no more than the force in any direction. A held edge resists the full force along its last motion and at least 58% of it in any other direction. The cube's turn about its diagonal is carried from step to step, so the frame does not flip as the motion turns. Before each physics step a trial solve checks the planned direction: the edge must come out held, or sliding within 8° of that direction. Otherwise the direction is re-aimed from the last two trials and tried again, up to five tries in all, and the best-aligned try is kept. The force applies at any speed. Friction only resists, so stiff soil stalls the arm against its joint force limits.
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
- The adhesion factor N<sub>ca</sub> is kept at zero or above, which the source does not do. Its negative values at small rakes would make the force negative, so the edge would cut for nothing: that happens at some rake for 90 of 1000 sampled soils. The floor acts below a rake of 0.37 rad in the nominal soil, the rake the demo cuts at, and below up to 0.7 rad in the least frictional soils.
- These values are assumptions, not taken from a source: the 2 cm edge thickness in the penetration term, the 45° dump angle, the 0.5 s pour time, the 0.3 m smallest spill area, the 175 kN force cap, treating loose spoil as cohesionless, and a repose angle equal to the friction angle. The force and the cut both use the 0.61 m edge of the bucket mesh, against 0.85 m for the bucket on the specification sheet, so the standard bucket would cut and resist about 40% more per pass.
- MuJoCo friction is a soft constraint, so a held edge still creeps. Resting in soil it creeps at up to about 1 mm/s, falling below the 0.44 mm/s creep of the joints alone within about 6 s. Against a stall in stiff soil it creeps at about 4 mm/s.
- An edge commanded slower than about 1.5 mm/s never loads its joints fully. At that scale each joint's speed control is soft too, so the arm pushes at about 30% of its force limit and the soil gives way at that load.
- Near a stall, the direction check often runs out of tries. The soil force can reverse the edge's motion within one physics step, so the planned and actual directions chase each other. Lowering the bucket teeth first about 1 m into nominal soil, about half of the moving steps end more than 8° off the motion, and the force does about 91% of the work it would do straight against it. Elsewhere it holds: across 80 combinations of joint commands, no moving step ended more than 8° off.
- Ground under the tracks is never cut. That does not fix [issue 6](../../issues/6): a trench edge just outside the tracks still gives too low a tipping load on a free base.

## Limits

- Simulation only. Nothing here has been checked against a real excavator.
- The soil is a height grid and a force formula, not grains.
- Calibrating the controller to a specific real machine is future work.

## Licence

The code is under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): you may read, run and change it for noncommercial purposes. Third-party files keep their own licences and are credited where they are added. The machine model is converted from [pwri-opera/zx120_ros](https://github.com/pwri-opera/zx120_ros) under Apache-2.0; its [notice](src/hma/machine/assets/NOTICE) lists the changes.

Required Notice: Copyright 2026 Gabriel Ezerzer
