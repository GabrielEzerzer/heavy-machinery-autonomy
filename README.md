# Heavy Machinery Autonomy

A simulation demo. You type a digging task in plain English, and a simulated 12-tonne excavator digs it. One controller, trained in simulation, drives every individual machine of this type with no retuning.

**Status: in development.** Nothing here runs yet. The [feature map](#feature-map) lists what is built and what is planned.

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
| Machine model | A 12 t class excavator with swing, boom, arm and bucket joints | Planned |
| Hydraulics model | Gives each machine its own delay, sluggishness, dead zone, lever-to-speed curve, flow limit, force limit and slack in the joints | Planned |
| Soil | A grid of ground heights that the bucket cuts, fills from and dumps onto, with a digging force that depends on the soil | Planned |
| Dig planner | Turns a trench into repeated cycles: cut, drag, curl, lift, swing, dump | Planned |
| Baseline controller | A conventional controller tuned once on the standard machine | Planned |
| Learned controller | One network that follows the planner's paths on any machine drawn | Planned |
| Training | One command that trains the controller in stages of rising difficulty | Planned |
| Evaluation | Scores a controller on machines it never saw in training | Planned |
| Language to task | A GPT model fills a task specification from a plain-English instruction | Planned |
| Plan checks | Refuses a specification that is malformed, out of reach or against trench safety rules, and asks a person to confirm | Planned |
| Demo screen | The plan, the drawn machine's card, both controllers side by side and the results chart | Planned |
| Balance on rough ground | The machine can slide and tip, and the controller learns to dig without unsettling it | Planned, after the first working demo |

Not planned: other makes, sizes or arm geometries; soil simulated as grains; buried obstacles.

## Limits

- Simulation only. Nothing here has been checked against a real excavator.
- The soil is a height grid and a force formula, not grains.
- Calibrating the controller to a specific real machine is future work.

## Licence

The code is under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): you may read, run and change it for noncommercial purposes. Third-party files keep their own licences and are credited where they are added.

Required Notice: Copyright 2026 Gabriel Ezerzer
