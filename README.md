# Heavy Machinery Autonomy

A simulation demo. You type a task in plain English, and a simulated 12-tonne excavator carries it out on rough ground without tipping over. One controller, trained in simulation, handles simulated machines of this type that each behave differently, with no retuning.

**Status: in development.** The machine model and its hydraulics are built, and a conventional controller follows bucket paths through them. The language step runs on its own and still produces trench plans from an earlier direction of the project. Rough ground, a carried load, the planner and the learned controller do not exist yet. An earlier version simulated soil and digging; it was removed and is in the repository's history. The [feature map](#feature-map) lists what is built and what is planned.

## How it works

1. A GPT model turns a plain-English instruction into a small task specification. The task type is being changed: today the specification still describes a trench, with a marker, a depth and a length.
2. Fixed checks accept or refuse the specification: valid fields, within the machine's reach, within the task's safety rules. A person confirms before anything moves.
3. A planner turns the specification into paths for the bucket and the load. It is not built yet.
4. A learned controller follows those paths. Every run draws a different machine, with its own hydraulic delay, dead zone, flow limit, slack in the joints and weights, and stands it on ground drawn at random. The controller is the same for all of them.

The physics is [MuJoCo](https://mujoco.org). The controller is trained with PPO from [Stable-Baselines3](https://stable-baselines3.readthedocs.io).

### Hydraulics model

`hma.hydraulics` sits between the controller and the joints. The controller sends four valve commands in [-1, 1] at the control rate (10 Hz by default; 20 Hz also works). At every 5 ms physics step, each command passes through a pure delay, a first-order lag, a dead zone, a command-to-speed curve with a gain, the joint's speed limit, a shared pump flow limit, a pump power limit and joint play, and reaches the joint as a speed. A joint's force limit is the friction cap of its valve in the MuJoCo model, so the joint stalls when its load exceeds the cap. The controller sees joint angles, joint speeds and a load signal (the force the valve passes to the joint, as a fraction of the cap). The load is positive when the valve pushes the joint in its positive direction, and reads exactly +1 or -1 while a joint is driven into an end stop or stalled. It falls to about 0 once a load above the cap has pushed the joint onto its end stop, because the stop then carries the load. All three readings arrive late and with noise added. One seed draws one machine, and the same seed always gives the same machine. Effects switch on in groups (`delay`, `valve`, `flow`, `limits`, `backlash`, `mass`, `sensors`) so that training can add them one step at a time; `flow` draws both pump flow and pump power. `NOMINAL` has a 0.05 s delay, a 0.1 s lag, no dead zone, a straight curve, full flow and power, nominal limits and no noise.

| Effect | Range per machine | Basis |
|---|---|---|
| Delay, per joint | 0.05–0.6 s | Up to 0.6 s in [Nan and Hutter][nh] |
| Lag after the delay | time constant 0.1–0.4 s | Assumed. The pilot-stage valves of [HEAP][heap] reach 90% of a commanded speed in 680–850 ms and pass about 0.5 Hz |
| Dead zone | 0–25% of full command | [Nan and Hutter][nh] |
| Command-to-speed curve | four segments with random positive slopes; gain ±10% | Random piecewise-linear curves with positive slopes in [Nan and Hutter][nh]; ±10% action scaling in [Egli and Hutter][eh] |
| Joint speed limit | nominal ±20%. Nominal: boom 0.3, arm 0.6, bucket 0.8, swing 0.8 rad/s | Boom, arm and bucket from Fig. 5 of [Egli et al.][e22], a 12 t machine. That paper does not drive swing, so the swing nominal is assumed, chosen so that the top of its range (0.96 rad/s) stays under the model's 1.0 rad/s actuator limit. The ±20% is assumed |
| Pump flow | 60–100% of a nominal supply that runs 2.5 joints at nominal full speed at once | Assumed. A real excavator slows its functions when two or three run together. Over 2,000 drawn machines with this supply, every set of three joints at full command shares flow on 99.65% of them, some pair shares on 61.6% and every pair on 28%. The nominal machine peaks at 1.6 joints' worth on the line path below, so it never shares there |
| Pump power | 85–100% of 56 kW delivered at the joints: a 70 kW engine times an assumed hydraulic efficiency of 0.8 | Hitachi lists 66.0 kW for the [ZX130-5G][zx5g] and 73.4 kW net for the [ZX130-5B][zx5b], 12–13 t machines of the class this model is converted from (unverified: the pages refuse automated reading). The efficiency and the range are assumed |
| Joint force limit | the model's cap ±15% | Assumed |
| Joint play | 0–0.5° | Assumed. [HEAP][heap] reports joint play but gives no size |
| Link masses, centres of mass, counterweight | ±10%; shifted by up to 5 cm; ±10% | Assumed |
| Sensor noise, uniform | ±0.02 rad, ±0.05 rad/s, ±5% of the force cap | Angle and speed noise from [Egli and Hutter][eh], who use 0.04–0.05 rad/s depending on the joint; the larger value is used for every joint here. Load noise is assumed |
| Sensor latency | 0–0.2 s | About 0.2 s of encoder-velocity delay in [Spinelli et al.][sp] |

Limits of the hydraulics model:

- Flow demand counts every joint alike: a joint at its nominal full speed uses one unit of flow, though real cylinders differ in size. When demand exceeds the pump, every moving joint slows by the same ratio.
- Power demand is each joint's latest force reading times its commanded speed, counted only while the joint pushes against its load. A boom lowering under gravity or a joint braking therefore draws nothing, where a real pump still spends some power pushing oil through the valve. A joint's share falls to zero smoothly as its force does, so no sign decision switches it on and off. The pump answers demand above the machine's power slowly, loosely as an engine's flywheel carries a short overload (an analogy, not a model of one): one ratio per machine slows every moving joint after flow sharing. Each second it rises by the fraction of the power left unused, or falls by the fraction overdrawn, divided by an assumed response time of 0.1 s. That makes the energy drawn above the limit roughly what the ratio's fall pays back, so over a window the average excess is about 0.1 s divided by the window, times how far the ratio fell. This is a rule of thumb, not a bound: the regulator uses the force read at the start of each step, and the force can change within the step. With every joint held at full command from reset on the nominal machine, power over 4.475 to 4.975 s averaged 1.000 times the limit while the ratio did not fall, and after the first 2 s the worst 0.5 s and 2 s averages were 1.08 and 1.02 times the limit (at most 1.07 and 1.03 on two drawn machines). The ratio stays exactly 1 on a run whose demand has stayed below the limit since reset. After an overload the slowdown lingers, because the ratio climbs back at the same rate: when the nominal machine's commands dropped from 1 to 0.1 after 15 s, once every joint was against its stop, the ratio took 0.41 s to pass 0.99, and 0.345 s after the drop it still slowed the joints by 17% although demand was already 0.89 times the limit. The limit is a coarse stand-in for a pump: commanded speed stands in for flow, and lowering or braking is free, so a learned controller may favour short bursts and gravity-assisted or braking motions more than a real machine would reward. The line and circle paths below are unchanged bit for bit on the nominal machine (peaks of 3.6 and 9.7 kW) and on 20 drawn machines, but the worst-corner machine described under Base controller, with the slowest valves, dips to a ratio of 0.88 on the line, moving its joints by up to 1.6e-3 rad. Sudden full commands can pass the limit briefly while the swing accelerates at its force cap: with random full commands every 0.1 s on the nominal machine (20 runs of 10 s), demand reached 1.28 times the power and the ratio fell to 0.92 before any joint came within 0.01 rad of an end stop. Once a joint sits on a stop at full force, demand reached 3.3 times the power and the ratio fell to 0.36. A slow regulator lets short excesses through. With every joint reversing at random each decision (10 or 20 Hz), or all joints or pairs of joints reversing together, against a load of 0.95 of its cap that follows the command or the joint's speed, pump power averaged over any 2 s stayed at or below 1.04 times the limit, but averaged over 0.5 s it reached 1.15 times the limit in the worst of 420 runs (42 machines, 10 patterns), so expect up to about 1.2. A joint pushing steadily near its force cap settles within 0.3% of the limit, and a joint driven into an end stop keeps commanding the speed the power allows at full force, which slows the other joints as a pump at relief pressure would.
- The curve's gain applies before the speed limit clips, so a machine with a gain below 1 tops out at its speed limit times its gain, and one with a gain above 1 reaches its limit early and holds it over the top of its command range (up to the last 10%). Both are plausible valve behaviour.
- Delay and lag are drawn independently. Over 500 drawn machines, the time for a joint to reach 90% of a commanded speed step has a median of 0.95 s and a maximum of 1.53 s, slower than the 680–850 ms measured on the [HEAP][heap] valves for 63% of joints: the ranges are meant to be wider than one measured machine. The nominal machine's 0.30 s is the fastest corner of every range.
- Joint play is lost motion when the command reverses. A link does not drop through the gap when its load reverses with no command.
- Delays and latency are rounded to the 5 ms physics step. The load reading is taken at the start of each step.
- Masses and force caps are written into the shared MuJoCo model for each machine by name, so a model cannot be shared by scenes that need different machines. Within one scene, repeatability is bit for bit only when the same machines are in it in the same states, because one constraint solve covers every machine: resetting one machine while another keeps running repeats its episode to about 1e-15: with both machines driven by random valve commands, the reset machine's joints differed by at most 1.1e-15 rad from a fresh scene. The scene clock (`data.time`) keeps running through a reset, so a controller that needs a bit-for-bit repeat should time its episode from its own step count.
- The machine model sets MuJoCo's solver tolerance to 1e-14. One constraint solve covers every machine in a scene, so a solver that stops early lets one machine's motion leak into another's. With one drawn machine driven by random valve commands for 20 s beside a drawn neighbour, compared with the same run with the neighbour still, the first machine's joints moved by up to 1.8e-15 rad at 1e-14 on welded bases, over six pairs of machines. On released bases four of the six pairs stayed at that level and two reached 3.5e-9 and 8.9e-9 rad, so the tolerance reduces the leak on a released base and does not remove it. The worst pair at 1e-12 moved 2.2e-8 rad welded and 1.6e-7 rad released, at 1e-10 3.4e-8 and 1.4e-6 rad, and at MuJoCo's default of 1e-8 2.0e-5 and 2.7e-4 rad. The tighter tolerance cost no speed beyond the spread between repeated runs. Scores are only comparable when taken in a fixed scene layout or with one machine per scene ([issue 11](../../issues/11)).
- MuJoCo makes each constraint softer the more easily the bodies move along it, a quantity it computes at the model's reference pose, here the home pose, when a scene is loaded and again whenever the hydraulics layer writes a machine's masses. The softness of the valves, joint limits and track contacts therefore follows the home pose and the drawn masses ([issue 23](../../issues/23)).
- A large swing step is limited by the swing's force cap as well as by the lag: the swing can accelerate at about 0.7 rad/s² at most.
- A drawn speed limit above the model's actuator limit is refused. For the arm that rules out more than 1.33 times its nominal speed.

**Throughput.** On an Intel Core i7-1185G7 laptop processor, in one process, with every machine on a welded base and given random valve commands at 10 Hz, a scene of one machine ran 9,400 to 11,000 physics steps per second through the hydraulics layer, about 50 times faster than real time, and a scene of eight machines 3,100 to 5,100 (25,000 to 41,000 machine steps per second), over repeated 20 s runs. MuJoCo's step took about a fifth of the time with one machine and about half with eight; the rest is the hydraulics layer, written in Python.

### Base controller

`BaseController` follows a bucket-tip path, given as position and pitch over time and turned into joint angles by inverse kinematics. Its command is the reference joint speed taken 0.17 s ahead, which covers the nominal machine's delay and lags, plus 0.25 /s times the joint angle error, divided by the nominal speed limit. It has no integral term. On the nominal machine it keeps the tip within 1 cm of a 3.1 m straight line driven in 4 s (mean 0.5 cm), at 10 Hz and at 20 Hz; with 0.5 s of delay the mean error grows to about 29 cm. The gain is set by the corner where every range is at its worst at once: 0.6 s delay, 0.4 s lag, 0.2 s sensor latency, curve gain 1.1, speed limits 1.2 times nominal, the steepest allowed first curve segment, full joint play and sensor noise. That machine moves up to 2.6 times faster per unit of command than the nominal one and responds about 1.25 s late, which puts the stability limit near 0.4 /s; 0.25 /s leaves a margin of about 1.6. At 1 /s the corner machine still swings by 1 to 4 m 8–10 s after the line ends, and 10 of 300 randomly drawn machines swing by more than 10 cm. At 0.25 /s no drawn machine does, and the corner machine without dead zones settles to within 1.4 cm in 30 s. The price is a large steady offset, because commands smaller than a joint's dead zone do nothing and there is no integral term to grow them. Over 300 randomly drawn machines at 10 Hz, 10 s after the line ends, the tip sits a median 133 cm and at worst 287 cm from its target. With the dead zones set to zero on the same machines, the median falls to 4 cm (at worst 66 cm), and the machines still moving keep converging without swinging.

[nh]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/cb538c33-9f47-4539-b238-2ef6372e62d2/content
[eh]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/b00c6c58-12a1-49af-a587-16872fe608a9/content
[e22]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/95ef5691-11e8-4a86-b02d-6f0e2501de9b/content
[heap]: https://arxiv.org/abs/2106.05059
[sp]: https://arxiv.org/abs/2410.05093
[zx5g]: https://www.hitachicm.com/ae/en/machinery/excavators/medium-excavators/product.zx130-5g
[zx5b]: https://www.hitachicm.com/us/en/products/excavators/medium-large/product.zx130-5b-la

## Feature map

A feature is **done** when it runs from a clean checkout and its checks pass.

| Feature | What it does | Status |
|---|---|---|
| Machine model | A 12 t class excavator with swing, boom, arm and bucket joints. Each joint takes a speed command and has a force limit. The base can be held still or left free to slide and tip | Done |
| Hydraulics model | Gives each machine its own delay, sluggishness, dead zone, lever-to-speed curve, flow limit, speed and force limits, slack in the joints, weights, and sensor noise and latency | Done |
| Baseline controller | A conventional controller tuned once on the standard machine | Done: follows paths in free air |
| Rough ground | Slopes, steps and edges of rigid ground. The machine can slide, rock and tip, and the arm and bucket meet the ground | Planned |
| Carried load | A load with real mass on the bucket | Planned |
| Planner | Turns a checked specification into paths for the bucket and the load | Planned |
| Cylinder forces | Joint force limits from the hydraulic cylinders' sizes and positions | Planned |
| Learned controller | One network that moves a load and props the machine on its bucket without tipping, on any machine drawn | Planned |
| Training | One command that trains the controller in stages of rising difficulty | Planned |
| Evaluation | Scores a controller on machines it never saw in training | Planned |
| Language to task | A GPT model fills a task specification from a plain-English instruction | Done for trench tasks; being changed to load handling |
| Plan checks | Refuses a specification that is malformed, out of reach or against trench safety rules, and asks a person to confirm | Done |
| Demo screen | The plan, the drawn machine's card, both controllers side by side and the results chart | Planned |

Not planned: digging and soil; other makes, sizes or arm geometries.

## Try it

Python 3.12 or newer.

```
pip install -e .
python -m hma.machine.demo
python -m hma.task "dig 1 m down at the far marker"
```

The first command opens a viewer and moves the arm through its range, with the hydraulic cylinders and bucket linkage drawn on their pins. The second asks a GPT model for a trench plan, the task type of the language step until it is changed, and checks it. It calls the [Codex CLI](https://github.com/openai/codex), which must be installed and signed in with your own ChatGPT account; this repo holds no API key.

## Limits

- Simulation only. Nothing here has been checked against a real excavator.
- Only the tracks touch the ground. The arm and bucket pass through it without force.
- The tracks meet the ground at separate contact points, not a continuous surface, so a track end that overhangs a drop tips the machine at too low a load ([issue 6](../../issues/6)).
- Calibrating the controller to a specific real machine is future work.

## Licence

The code is under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): you may read, run and change it for noncommercial purposes. Third-party files keep their own licences and are credited where they are added. The machine model is converted from [pwri-opera/zx120_ros](https://github.com/pwri-opera/zx120_ros) under Apache-2.0; its [notice](src/hma/machine/assets/NOTICE) lists the changes.

Required Notice: Copyright 2026 Gabriel Ezerzer
