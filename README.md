# Heavy Machinery Autonomy

A simulation demo. You type a digging task in plain English, and a simulated 12-tonne excavator digs it. One controller, trained in simulation, drives every individual machine of this type with no retuning.

**Status: in development.** The machine model, the soil and the language step run on their own; a scripted dig cycle runs, but no controller digs yet. The [feature map](#feature-map) lists what is built and what is planned.

## How it works

1. A GPT model turns an instruction such as "dig 1 m down at the far marker" into a small task specification: which marker, how deep, how long.
2. Fixed checks accept or refuse the specification: valid fields, within the machine's reach, within trench safety rules. A person confirms before anything moves.
3. A dig planner turns the specification into paths for the bucket.
4. A learned controller follows those paths. Every run draws a different machine, with its own hydraulic delay, dead zone, flow limit, slack in the joints, weights and soil. The controller is the same for all of them.

The physics is [MuJoCo](https://mujoco.org). The soil is a grid of heights with an analytic digging force. The controller is trained with PPO from [Stable-Baselines3](https://stable-baselines3.readthedocs.io).

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
- Power demand is each joint's latest force reading times its commanded speed, counted only while the joint pushes against its load. A boom lowering under gravity or a joint braking therefore draws nothing, where a real pump still spends some power pushing oil through the valve. A joint's share falls to zero smoothly as its force does, so no sign decision switches it on and off. The pump answers demand above the machine's power slowly, loosely as an engine's flywheel carries a short overload (an analogy, not a model of one): one ratio per machine slows every moving joint after flow sharing. Each second it rises by the fraction of the power left unused, or falls by the fraction overdrawn, divided by an assumed response time of 0.1 s. That makes the energy drawn above the limit roughly what the ratio's fall pays back, so over a window the average excess is about 0.1 s divided by the window, times how far the ratio fell. This is a rule of thumb, not a bound: the regulator uses the force read at the start of each step, and the force can change within the step. With every joint held at full command from reset on the nominal machine, power over 4.475 to 4.975 s averaged 1.007 times the limit while the ratio did not fall, and after the first 2 s the worst 0.5 s and 2 s averages were 1.08 and 1.02 times the limit (at most 1.08 and 1.04 on two drawn machines). The ratio stays exactly 1 on a run whose demand has stayed below the limit since reset. After an overload the slowdown lingers, because the ratio climbs back at the same rate: when the nominal machine's commands dropped from 1 to 0.1 after 6 s against its stops, the ratio took 0.41 s to pass 0.99, and 0.345 s after the drop it still slowed the joints by 17% although demand was already 0.89 times the limit. The limit is a coarse stand-in for a pump: commanded speed stands in for flow, and lowering or braking is free, so a learned controller may favour short bursts and gravity-assisted or braking motions more than a real machine would reward. The line and circle paths below are unchanged bit for bit on the nominal machine (peaks of 3.6 and 9.7 kW) and on 20 drawn machines, but the worst-corner machine described under Base controller, with the slowest valves, dips to a ratio of 0.88 on the line, moving its joints by up to 1.6e-3 rad. Sudden full commands can pass the limit briefly while the swing accelerates at its force cap: with random full commands every 0.1 s on the nominal machine (20 runs of 10 s), demand reached 1.28 times the power and the ratio fell to 0.88 before any joint came within 0.01 rad of an end stop. Once a joint sits on a stop at full force, demand reached 3.2 times the power and the ratio fell to 0.41. A slow regulator lets short excesses through. With every joint reversing at random each decision (10 or 20 Hz), or all joints or pairs of joints reversing together, against a load of 0.95 of its cap that follows the command or the joint's speed, pump power averaged over any 2 s stayed at or below 1.04 times the limit, but averaged over 0.5 s it reached 1.15 times the limit in the worst of 420 runs (42 machines, 10 patterns), so expect up to about 1.2. A joint pushing steadily near its force cap settles within 0.3% of the limit, and a joint driven into an end stop keeps commanding the speed the power allows at full force, which slows the other joints as a pump at relief pressure would.
- The curve's gain applies before the speed limit clips, so a machine with a gain below 1 tops out at its speed limit times its gain, and one with a gain above 1 reaches its limit early and holds it over the top of its command range (up to the last 10%). Both are plausible valve behaviour.
- Delay and lag are drawn independently. Over 500 drawn machines, the time for a joint to reach 90% of a commanded speed step has a median of 0.95 s and a maximum of 1.53 s, slower than the 680–850 ms measured on the [HEAP][heap] valves for 63% of joints: the ranges are meant to be wider than one measured machine. The nominal machine's 0.30 s is the fastest corner of every range.
- Joint play is lost motion when the command reverses. A link does not drop through the gap when its load reverses with no command.
- Delays and latency are rounded to the 5 ms physics step. The load reading is taken at the start of each step.
- Masses and force caps are written into the shared MuJoCo model for each machine by name, so a model cannot be shared by scenes that need different machines. Within one scene, repeatability is bit for bit only when the same machines are in it in the same states, because one constraint solve covers every machine: resetting one machine while another keeps running repeats its episode to about 1e-14. The scene clock (`data.time`) keeps running through a reset, so a controller that needs a bit-for-bit repeat should time its episode from its own step count.
- A large swing step is limited by the swing's force cap as well as by the lag: the swing can accelerate at about 0.7 rad/s² at most.
- A drawn speed limit above the model's actuator limit is refused. For the arm that rules out more than 1.33 times its nominal speed.

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
| Soil | A grid of ground heights that the bucket cuts, fills from and dumps onto, with a digging force that depends on the soil | Done |
| Dig planner | Turns a trench into repeated cycles: cut, drag, curl, lift, swing, dump | Planned |
| Baseline controller | A conventional controller tuned once on the standard machine | Done for free-air paths; dig cycles wait for the soil and the planner |
| Learned controller | One network that follows the planner's paths on any machine drawn | Planned |
| Training | One command that trains the controller in stages of rising difficulty | Planned |
| Evaluation | Scores a controller on machines it never saw in training | Planned |
| Language to task | A GPT model fills a task specification from a plain-English instruction | Done |
| Plan checks | Refuses a specification that is malformed, out of reach or against trench safety rules, and asks a person to confirm | Done |
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
- **Spoil.** Cohesion, adhesion and penetration act only on the part of the cut below the loose spoil. Spoil adds its weight and surcharge but no cohesion, so recutting a heap costs far less than cutting the bank: in the stiffest soil a 0.25 m drag meets 134 kN in undisturbed ground and about 2 kN in spoil. The split is taken at each point along the edge, and the deepest ground and the deepest undisturbed ground are found separately, so the force changes smoothly where spoil meets undisturbed ground and as a heap rises past the edge.
- **How the force acts.** Three MuJoCo friction-loss tendons run from the bucket tip along axes set by the bucket's pose alone: h, horizontal and across the edge; s, horizontal and along the edge; v, vertical. Before every physics step each axis gets its own limit from the pose and the ground, never from the motion or the command:
  - h: the full separation term plus the penetration term times the horizontal part of the lip, for the deepest ground one cell ahead of, one cell behind and at each point of the edge. The rake is measured from a horizontal direction fixed to the edge, so it changes continuously as the bucket curls past an upright opening.
  - s: the same, for the deepest ground at each point of the edge and one cell beyond each end. On level ground a sideways sweep costs the same as cutting forward at that depth, and a ridge narrower than the edge still resists it.
  - v: the penetration term alone, for the undisturbed ground at the points of the edge, capped smoothly at 12 kN. It is zero in spoil. The cap sits below the smallest upward force the tip can exert while holding a full bucket (9.9 kN of soil) anywhere in the digging envelope, with every joint limited to 0.85 of its nominal force, the lowest the hydraulics model draws. That force was found statically, as the largest upward tip force the boom, arm and bucket joints can each add to gravity within their limits, over 13,651 poses with the tip 3.5 to 9 m from the swing axis and from 1.5 m below to 0.3 m above the ground. Its minimum is 15.4 kN, at full reach, so the cap has a 22% margin.

  All three are zero when no ground around the edge is above it. Friction only resists, so the soil never pushes, and stiff soil stalls the arm against its joint force limits. The machine model sets MuJoCo's solver tolerance to 1e-12. At 1e-10 a machine given random commands moved a stalled arm on another machine in the same simulation by up to 2.3e-7 rad, against 3e-15 rad at 1e-12, at no measurable cost in speed. At the default the solver can stop after one iteration where the edge starts or stops slipping.
- **Stick and slip.** Each axis is Coulomb friction: when the edge stops slipping, the force on that axis falls from its limit to what holds the bucket still, and it returns to the limit when the edge slips again. Holding the boom fully down for 16 s, the largest change in an axis force between two physics steps is 2.9 kN in the nominal soil, 4.4 kN in the stiffest and 6.9 kN in a soil with no cohesion or penetration resistance. With the commands changed at random at 10 or 20 Hz, it is 2.0 kN. Releasing an arm stalled in the stiffest soil drops the force by 11.5 kN in one step. These figures cover single descents, one stall and its release. With random commands on every joint the steps are larger: in 48 runs of 30 s, 22 physics steps in 16 runs change the soil force by more than 30 kN, the largest by 93 kN.
- **Against the published forces.** A 0.3 m drag through the nominal soil meets about 48 kN on the h axis with an empty bucket and about 56 kN with a full one, and the demo's drag a median of 52 kN. These are below the arm crowd force of 63 kN (SAE) to 65 kN (ISO), and the bucket digging force of 86 kN (SAE) to 99 kN (ISO), on the specification sheet. The stiffest soil, at about 149 kN, stalls the arm.
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
- The adhesion factor N<sub>ca</sub> is kept at zero or above, which the source does not do. Its negative values at small rakes would make the force negative, so the edge would cut for nothing. For a 0.3 m cut that happens at some rake for 69 of 1000 sampled soils with the penetration term, and for 334 without it; at some depth between 2 cm and 0.5 m, for 97 with it and 528 without. The floor acts below a rake of 0.37 rad in the nominal soil, the rake the demo cuts at, and below up to 0.63 rad in the least frictional soils.
- These values are assumptions, not taken from a source: the 2 cm edge thickness in the penetration term, the 45° dump angle, the 0.5 s pour time, the 0.3 m smallest spill area, the 175 kN force cap, the 12 kN cap on the vertical axis, treating loose spoil as cohesionless, and a repose angle equal to the friction angle. The force and the cut both use the 0.61 m edge of the bucket mesh, against 0.85 m for the bucket on the specification sheet, so the standard bucket would cut and resist about 40% more per pass.
- MuJoCo friction is a soft constraint, so a held edge still creeps: at about 0.23 mm/s resting in soil. With the arm pushing at its force limit into the stiffest soil 0.25 m deep, the edge creeps forward at about 4 mm/s with the h force at 80 kN, half its 159 kN limit, and sinks at about 8.5 mm/s with the v axis at its cap.
- Each joint's speed control is soft at small speeds, so a slow edge never loads its joints fully. Pushing into the stiffest soil, an arm command that moves the edge at about 1.5 mm/s loads the arm to 22% of its force limit, and one at about 5 mm/s to 58%; the soil gives way at those loads.
- The force is set by the pose, not the motion, which costs fidelity in these ways. The numbers are for the nominal soil unless stated.
  - Each axis resists equally both ways, so backing away from a cut face costs about as much as cutting forward. From a face priced at 65 kN, backing away 5, 10 and 20 cm takes 1.05, 0.81 and 0.52 times the work of cutting forward that far, at up to 66 kN.
  - Moving the back of the bucket first is priced as if the lip led. There is no back-of-blade rake.
  - Diagonal motion pays each axis it moves along, so the total force can reach the root of the sum of the squared limits. Over the demo cycle it reaches 1.25 times the h limit, and 1.42 times over 30 random soils.
  - The soil holds a bucket at rest up to the limits on all axes, but vertically only up to the 12 kN cap. With the boom held fully down, the edge is about 3 m deep after 1.5 s, and after 5 s it is 3.4, 3.0 and 2.9 m deep in a soil with no cohesion or penetration resistance, the nominal soil and the stiffest. Lifting out pays the same cap: with the joints driven at three times their error from `inverse_kinematics` at a pitch of 1.9 rad, towards a target rising at 0.3 m/s, the edge leaves a 0.5 m slot in 2.1 s for 0.02 kJ in the nominal soil and 0.08 kJ in the stiffest, at up to 12 and 19 kN.
  - A deep bucket needs the right escape. After 1.5 to 5 s of full boom-down, raising the boom alone or pushing the arm out alone leaves the bucket in the ground after 15 s in all 30 random soils; raising the boom while curling frees it in all 30, in about 1.1 s in the nominal soil. From 0.5 m deep after 3 s of the arm crowding in, pushing the arm out alone leaves it in the ground in 22 of 30 soils, and every other escape frees it. From drags 0.2 and 0.4 m deep, every escape frees it except pushing the arm out at 0.4 m in one soil.
  - Raising the boom eases the vertical hold, which lets a bucket pull harder horizontally. In the stiffest soil with the arm crowding in, the steady pull is about 82 kN, and raising the boom slowly adds up to 12 kN; faster lifts the bucket out.
  - In 7 of 30 random soils (`sample_soil` seeds 2, 4, 7, 9, 13, 14 and 21) the demo's drag runs to its 15 s timeout, where the h limit peaks at 76 to 105 kN, against at most 70 kN in the soils that finish. No curl, lift or swing phase times out, and all 30 complete a dump.
- A cell is lowered when the edge passes its centre, not by the area the edge sweeps over it, and this can be exploited repeatably. Plunging the edge onto a row of cell centres and nudging it 1 cm along its travel cuts the whole row. Repeated row by row, that moves 5.7, 4.2 and 2.6 times as much soil per kilojoule as a drag at 3, 5 and 10 cm deep in the nominal soil, and 5.6, 4.7 and 4.2 times in the stiffest. Each crossing takes one row of cells, about 15 litres at 0.25 m deep and 60 litres at 1 m. A reward must not pay for volume cut until cutting follows the area swept.
- Ground under the tracks is never cut. That does not fix [issue 6](../../issues/6): a trench edge just outside the tracks still gives too low a tipping load on a free base.

## Limits

- Simulation only. Nothing here has been checked against a real excavator.
- The soil is a height grid and a force formula, not grains.
- Calibrating the controller to a specific real machine is future work.

## Licence

The code is under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): you may read, run and change it for noncommercial purposes. Third-party files keep their own licences and are credited where they are added. The machine model is converted from [pwri-opera/zx120_ros](https://github.com/pwri-opera/zx120_ros) under Apache-2.0; its [notice](src/hma/machine/assets/NOTICE) lists the changes.

Required Notice: Copyright 2026 Gabriel Ezerzer
