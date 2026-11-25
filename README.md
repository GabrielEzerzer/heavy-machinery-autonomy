# Heavy Machinery Autonomy

A simulation demo. You type a task in plain English, and a simulated 12-tonne excavator carries it out on rough ground without tipping over. One controller, trained in simulation, handles simulated machines of this type that each behave differently, with no retuning.

**Status: in development.** The machine model, its hydraulics and its tracks on rough rigid ground are built, and a conventional controller follows bucket paths through them. The language step runs on its own and still produces trench plans from an earlier direction of the project. Contact between the arm or bucket and the ground, a carried load, the planner and the learned controller do not exist yet. An earlier version simulated soil and digging; it was removed and is in the repository's history. The [feature map](#feature-map) lists what is built and what is planned.

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
| Lag after the delay | time constant 0.1–0.4 s | Assumed. The pilot-stage valves of [HEAP][heap] reach 90% of a commanded speed in 680–850 ms and pass about 0.5 Hz. The delay range does not already contain this lag: [Nan and Hutter][nh] model the pilot stage as the pure delay and the main stage as separate second-order oil-compressibility dynamics, which this lag stands in for |
| Dead zone | 0–25% of full command | [Nan and Hutter][nh] |
| Command-to-speed curve | four segments with random positive slopes; gain ±10% | Random piecewise-linear curves with positive slopes in [Nan and Hutter][nh]; ±10% action scaling in [Egli and Hutter][eh] |
| Joint speed limit | nominal ±20%. Nominal: boom 0.3, arm 0.6, bucket 0.8, swing 0.8 rad/s | Boom, arm and bucket from Fig. 5 of [Egli et al.][e22], a 12 t machine. That paper does not drive swing, so the swing nominal is assumed, chosen so that the top of its range (0.96 rad/s) stays under the model's 1.0 rad/s actuator limit. The [ZX120 spec sheet][zx120] gives 13.7 rpm (1.43 rad/s), so the model's swing is slow. The ±20% is assumed |
| Pump flow | 60–100% of a nominal supply that runs 2.5 joints at nominal full speed at once | Assumed. A real excavator slows its functions when two or three run together. Over 2,000 drawn machines with this supply, every set of three joints at full command shares flow on 99.65% of them, some pair shares on 61.6% and every pair on 28%. The nominal machine peaks at 1.6 joints' worth on the line path below, so it never shares there |
| Pump power | 85–100% of 56 kW delivered at the joints: a 70 kW engine times an assumed hydraulic efficiency of 0.8 | The [ZX120 spec sheet][zx120] gives 65 kW net (ISO 9249), so 70 kW is about 8% high. Hitachi lists 66.0 kW for the [ZX130-5G][zx5g] and 73.4 kW net for the [ZX130-5B][zx5b] (unverified: the pages refuse automated reading). The efficiency and the range are assumed |
| Joint force limit | the model's cap ±15% | Assumed. The cylinders give limits that change with pose and direction instead; see [Cylinders](#cylinders) |
| Joint play | 0–0.5° | Assumed. [HEAP][heap] reports joint play but gives no size |
| Link masses, centres of mass, counterweight | ±10%; shifted by up to 5 cm; ±10% | Assumed |
| Sensor noise, uniform | ±0.02 rad, ±0.05 rad/s, ±5% of the force cap | Angle and speed noise from [Egli and Hutter][eh], who use 0.04–0.05 rad/s depending on the joint; the larger value is used for every joint here. Load noise is assumed |
| Sensor latency | 0–0.2 s | About 0.2 s of encoder-velocity delay in [Spinelli et al.][sp] |

Limits of the hydraulics model:

- Flow demand counts every joint alike: a joint at its nominal full speed uses one unit of flow, though real cylinders differ in size. When demand exceeds the pump, every moving joint slows by the same ratio.
- Power demand is each joint's latest force reading times its commanded speed, counted only while the joint pushes against its load. A boom lowering under gravity or a joint braking therefore draws nothing, where a real pump still spends some power pushing oil through the valve. A joint's share falls to zero smoothly as its force does, so no sign decision switches it on and off. The pump answers demand above the machine's power slowly, loosely as an engine's flywheel carries a short overload (an analogy, not a model of one): one ratio per machine slows every moving joint after flow sharing. Each second it rises by the fraction of the power left unused, or falls by the fraction overdrawn, divided by an assumed response time of 0.1 s. That makes the energy drawn above the limit roughly what the ratio's fall pays back, so over a window the average excess is about 0.1 s divided by the window, times how far the ratio fell. This is a rule of thumb, not a bound: the regulator uses the force read at the start of each step, and the force can change within the step. With every joint held at full command from reset on the nominal machine, power over 4.475 to 4.975 s averaged 1.000 times the limit while the ratio did not fall, and after the first 2 s the worst 0.5 s and 2 s averages were 1.08 and 1.02 times the limit (at most 1.07 and 1.03 on two drawn machines). The ratio stays exactly 1 on a run whose demand has stayed below the limit since reset. After an overload the slowdown lingers, because the ratio climbs back at the same rate: when the nominal machine's commands dropped from 1 to 0.1 after 15 s, once every joint was against its stop, the ratio took 0.41 s to pass 0.99, and 0.345 s after the drop it still slowed the joints by 17% although demand was already 0.89 times the limit. The limit is a coarse stand-in for a pump: commanded speed stands in for flow, and lowering or braking is free, so a learned controller may favour short bursts and gravity-assisted or braking motions more than a real machine would reward. The line and circle paths below are unchanged bit for bit on the nominal machine (peaks of 3.6 and 9.7 kW) and on 20 drawn machines, but the worst-corner machine described under Base controller, with the slowest valves, dips to a ratio of 0.88 on the line, moving its joints by up to 1.6e-3 rad. Sudden full commands can pass the limit briefly while the swing accelerates at its force cap: with random full commands every 0.1 s on the nominal machine (20 runs of 10 s), demand reached 1.28 times the power and the ratio fell to 0.92 before any joint came within 0.01 rad of an end stop. Once a joint sits on a stop at full force, demand reached 3.3 times the power and the ratio fell to 0.36. A slow regulator lets short excesses through. With every joint reversing at random each decision (10 or 20 Hz), or all joints or pairs of joints reversing together, against a load of 0.95 of its cap that follows the command or the joint's speed, pump power averaged over any 2 s stayed at or below 1.04 times the limit, but averaged over 0.5 s it reached 1.15 times the limit in the worst of 420 runs (42 machines, 10 patterns), so expect up to about 1.2. A joint pushing steadily near its force cap settles within 0.3% of the limit, and a joint driven into an end stop keeps commanding the speed the power allows at full force, which slows the other joints as a pump at relief pressure would.
- The curve's gain applies before the speed limit clips, so a machine with a gain below 1 tops out at its speed limit times its gain, and one with a gain above 1 reaches its limit early and holds it over the top of its command range (up to the last 10%). Both are plausible valve behaviour.
- Delay and lag are drawn independently. Over 500 drawn machines, the time for a joint to reach 90% of a commanded speed step has a median of 0.95 s and a maximum of 1.53 s, slower than the 680–850 ms measured on the [HEAP][heap] valves for 63% of joints: the ranges are meant to be wider than one measured machine. The nominal machine's 0.30 s is the fastest corner of every range.
- Joint play is lost motion when the command reverses. A link does not drop through the gap when its load reverses with no command, which is when a bucket touching down takes the machine's weight. The play state also ignores motion forced by the load: with the arm commanded at +1 against 1.5 times its cap for 0.2 s, the arm was pushed back 30° while the play state went to the + end.
- Each joint's MuJoCo valve adds a lag of 20 ms (valve inertia 1e8 over gain 5e9) in series with the drawn lag.
- With zero command, a joint under a static load creeps, because the valve holds it through a soft friction constraint: the boom on the nominal machine sank 0.0075°/s at half its cap and 0.014°/s at 0.95 of it. A real boom or arm is held by load-holding valves and drifts millimetres over minutes.
- The pump power regulator counts a joint's power while the joint crosses its play with no output, and it slows every joint by the same ratio, including a boom lowering under gravity, which a real machine does by regeneration and meter-out rather than pump power.
- The load reading is a signed fraction of the machine's own force limit, so the reading at rest gives that hidden limit away: the boom reads -0.212 at the home pose at 0.85 times the cap and -0.157 at 1.15 times. A real machine measures two non-negative chamber pressures per cylinder.
- There is no oil compressibility: after a sudden stop from -0.3 rad/s the boom stops without bouncing.
- Delays and latency are rounded to the 5 ms physics step. The load reading is taken at the start of each step.
- Masses and force caps are written into the shared MuJoCo model for each machine by name, so a model cannot be shared by scenes that need different machines. Within one scene, repeatability is bit for bit only when the same machines are in it in the same states, because one constraint solve covers every machine: resetting one machine while another keeps running repeats its episode to about 1e-15: with both machines driven by random valve commands, the reset machine's joints differed by at most 1.1e-15 rad from a fresh scene. The scene clock (`data.time`) keeps running through a reset, so a controller that needs a bit-for-bit repeat should time its episode from its own step count.
- The machine model sets MuJoCo's solver tolerance to 1e-14. One constraint solve covers every machine in a scene, so a solver that stops early lets one machine's motion leak into another's. With one drawn machine driven by random valve commands for 20 s beside a drawn neighbour, compared with the same run with the neighbour still, the first machine's joints moved by up to 1.8e-15 rad at 1e-14 on welded bases, over six pairs of machines. On released bases four of the six pairs stayed at that level and two reached 3.5e-9 and 8.9e-9 rad, so the tolerance reduces the leak on a released base and does not remove it. The worst pair at 1e-12 moved 2.2e-8 rad welded and 1.6e-7 rad released, at 1e-10 3.4e-8 and 1.4e-6 rad, and at MuJoCo's default of 1e-8 2.0e-5 and 2.7e-4 rad. The tighter tolerance cost no speed beyond the spread between repeated runs. Scores are only comparable when taken in a fixed scene layout or with one machine per scene ([issue 11](../../issues/11)).
- MuJoCo makes each constraint softer the more easily the bodies move along it, a quantity it computes at the model's reference pose, here the home pose, when a scene is loaded and again whenever the hydraulics layer writes a machine's masses. The softness of the valves, joint limits and track contacts therefore follows the home pose and the drawn masses ([issue 23](../../issues/23)).
- A large swing step is limited by the swing's force cap as well as by the lag: the swing can accelerate at about 0.7 rad/s² at most.
- A drawn speed limit above the model's actuator limit is refused. For the arm that rules out more than 1.33 times its nominal speed.

**Throughput.** On an Intel Core i7-1185G7 laptop processor, in one process, each machine got random valve commands at 10 Hz, and random travel commands when travelling. Each figure covers three 20 s runs (`python -m hma.machine.measure throughput`), in physics steps per second through the hydraulics layer:

| Scene | Welded base | Released, standing | Released, travelling |
|---|---|---|---|
| 1 machine | 6,500–7,800 | 1,130–1,160 | 900–1,080 |
| 8 machines | 3,000–3,500 | 380–410 | 370–374 |

In the same session the model before tracks ran welded at 6,300–10,000 steps per second for one machine and 3,600–4,700 for eight. On an earlier day it ran 9,400–11,000 and 3,100–5,100, so the laptop's speed varies from day to day. The belt joints and travel valves cost about a fifth on eight welded machines.

A released machine runs about six times slower than a welded one, because its track contacts are found in Python every step ([issue 28](../../issues/28)). That takes about 0.65 ms per step for one machine, where MuJoCo's own step takes about 15 µs.

### Cylinders

`hma.machine.cylinders` holds the boom, arm and bucket cylinders: their mounting pins in the model's link frames, their sizes and relief pressures, and each cylinder's length and lever arm (the rate of change of length with joint angle) as functions of the joint angle. The bucket cylinder works through the bucket's four-bar linkage, so its lever arm varies most. `python -m hma.machine.pins <zx120_ros checkout>` rebuilds the pin table from the source model: pins from its `zx120.xacro` before commit 31e8ae6 (the cylinder joints were later removed), the two rod-end pins and both link lengths fitted on its meshes. `python -m hma.machine.statics` prints the comparisons below. **The simulation does not use these limits yet**: joints still stall at the constant caps in the table above, and the mechanism that would apply them is open work.

| Cylinder | Count, bore, rod | Head / rod area | Stroke over the joint range | Lever arm over the range |
|---|---|---|---|---|
| Boom | 2 × 105 mm, 70 mm | 173.2 / 96.2 cm² (1.80) | 1.506–2.459 m (0.953 m) | 0.272–0.571 m; extending raises |
| Arm | 1 × 115 mm, 80 mm | 103.9 / 53.6 cm² (1.94) | 1.652–2.811 m (1.159 m) | 0.279–0.666 m; extending crowds |
| Bucket | 1 × 100 mm, 70 mm | 78.5 / 40.1 cm² (1.96) | 1.358–2.213 m (0.855 m) | 0.096–0.391 m; extending curls |

Sizes and the main relief pressure, 34.3 MPa, are from the [ZX120 spec sheet][zx120]. The overload (port) reliefs, 37.3 MPa on the boom, arm crowd and bucket curl sides and 39.2 MPa on arm out and bucket open, and the main relief's tolerance of +2.0/−0.5 MPa are from a repair manual's listing of the ZX110 to ZX135 technical manual ([listing][zxman], unverified against the manual). Strokes are not published; each range's retracted length exceeds its stroke by 0.49 to 0.55 m, which fits one cylinder.

A joint's torque limit is the cylinder's force at a relief pressure times its lever arm: pressure × head area pushing, pressure × rod area pulling. The two directions differ by the area ratio, and both vary with the lever arm. Moving one pin 1 cm in its worst direction changes the lever arm, and the torque limit with it, by at most 1.29 cm on the boom (3.5% at 44°), 1.33 cm on the arm (3.4% at 30°) and 1.15 cm on the bucket (6.1% at full open, 4.7% at full curl). A 1 cm error in either bucket link length changes the bucket lever arm by up to 2.8 to 4.7%, depending on the link and the sign of the error. A wrong overload relief setting changes only how hard a held joint resists before it gives way, by up to 14% over the main relief. At main relief, in kN·m, with the positive direction lowering the boom, crowding the arm and curling the bucket:

| Joint | Angle | + direction | − direction | Old cap, both ways |
|---|---|---|---|---|
| Boom | −70°, −32°, 6°, 44° | 90, 163, 187, 122 (rod side) | 162, 294, 337, 220 (head side) | 330 |
| Arm | 30°, 71°, 111°, 152° | 140, 236, 201, 99 | 72, 122, 104, 51 | 244 |
| Bucket | −33°, 26°, 84°, 143° | 50, 105, 69, 26 | 26, 54, 35, 13 | 122 |

The old caps overstate pushing the boom down, as in propping the machine, by 1.8 to 3.7 times, and opening the bucket near full curl by 9 times.

Comparisons with the spec sheet, from `python -m hma.machine.statics`:

- **Digging forces** ([ISO 6015][iso6015]: the force at the bucket teeth at the position of greatest moment, at relief pressure). Bucket: 105.4 kN·m at the model's 1.230 m tip radius is 85.7 kN, against 86 kN (SAE) and 99 kN (ISO) on the spec sheet; 99 kN would need a radius of 1.064 m. Arm: 237.1 kN·m over 3.680 m from the arm pivot to the tip is 64.4 kN, against 65 kN (ISO) and 63 kN (SAE). The spec sheet does not say what makes its ISO and SAE figures differ.
- **Lift capacity**, without bucket, 2.52 m arm, at the bucket pin. The chart's ratings are 87% of hydraulic capacity or 75% of tipping load, whichever is less (ISO 10567). Where the chart marks a rating as hydraulically limited, 87% of the model's hydraulic capacity is within +0.6% to +5.5% of it: 3,425 against 3,350 kg (4.5 m high, 4.5 m out), 5,507 against 5,460 (3.0, 3.0), 4,138 against 4,070 (3.0, 4.5), 7,941 against 7,890 (1.5, 3.0), 7,044 against 6,730 (ground, 3.0), 4,883 against 4,630 (−1.5, 1.5), 8,851 against 8,540 (−3.0, 1.5) and 7,526 against 7,340 (−3.0, 3.0). Boom raising limits most points; the arm limits the three close to the machine, with its cylinder at main relief.
- **Tipping**, for the undercarriage: 75% of the model's tipping load is within −2.7% to +2.1% of every tipping-limited rating on the chart, over the front and over the side. [Machine on the ground](#machine-on-the-ground) lists each entry. The tipping lines come from the track model: x = ±1.44 m through the idler and sprocket centres, and y = ±1.065 m along the line where the rollers meet the track links ([ISO 10567][iso10567]). The masses are those in the model, 11,600 kg without the bucket. The two points 1.5 m out lie inside the front tipping line, so no load there tips the machine forward.

**Propping.** Pushing the boom down with the bucket on level ground ahead of the tracks lifts the front of the tracks wherever the tip is 2.5 to 8 m ahead of the swing centre, with the boom at main relief and the arm and bucket held at main relief. At the best pose for each reach the joints hold 1.29 times the tip force that lifts the front at 2.5 m (arm-limited), 2.03 times at 3 m (bucket-limited), 2.09 times at 3.5 m (arm-limited) and 1.66 to 1.90 times from 4 to 8 m (boom-limited). For example, at 5 m the joints hold 52.7 kN and lifting needs 29.4 kN. Holding the arm and bucket at their overload reliefs instead raises the margin by at most 0.17 close in and leaves it unchanged from 4 m out. The old caps gave margins of 2.5 to 3.7 times. The lifting force is checked in simulation with an upward force at the bucket tip ([Machine on the ground](#machine-on-the-ground)); the arm and bucket do not touch the ground in the model yet.

Left out, with what each would change:

- **The mechanism.** The joints still stall at constant symmetric caps. Applying these limits needs a valve model that also gives load-holding, pressure build-up and the asymmetric speeds below, and the friction-tendon valve cannot give those.
- **Speed asymmetry.** For one flow, a cylinder retracts 1.8 to 2 times faster than it extends, and joint speed varies with the lever arm. Both directions share one speed limit here. The machine's boom and arm regeneration is not modelled either.
- **End-of-stroke cushions.** The joint limits stand in for the cylinder end stops, with no cushioning.
- **Geometry.** The source model's two versions place the boom pivot 11 cm apart; the pins follow the meshes, which sit 1.5 to 4.5 cm from the model's joints. The bucket's H-link length, fitted on its mesh at 0.4656 m, is uncertain by about 5 mm.
- **Rated pressures only.** Cylinder friction, seal drag and the back pressure of the return oil, which lower the real force, are not modelled.

### Machine on the ground

`hma.machine.mass` builds each body's mass, centre of mass and inertia from its parts. `hma.machine.tracks` stands the machine on two driven tracks and replaces `mujoco.mj_step` with `tracks.step`. `hma.machine.ground` draws rough rigid ground from a seed. `python -m hma.machine.measure tipping` and `python -m hma.machine.measure throughput` print the measurements below.

**Mass.** The model weighs the published operating weight, 12,000 kg, with 500 mm shoes, the 2.52 m arm, a 0.45 m³ bucket and the 2,450 kg counterweight ([spec sheet][zx120]).

| Part | Mass, kg | Source |
|---|---|---|
| Counterweight; boom with arm cylinder; arm with bucket cylinder; bucket | 2,450; 1,220; 600; 400 | [ZX120 spec sheet][zx120] |
| Fuel, 250 L | 205 | Tank size from the [spec sheet][zx120]; diesel at 0.82 kg/L, assumed |
| Operator | 75 | Assumed, the usual operating-weight convention |
| Track chains with 500 mm shoes, both sides | 1,540 | [Cat 313 GC spec sheet][cat313], a 13 t machine with the same 1,990 mm gauge and shoe width |
| Two boom cylinders | 240 | [Cat 313 GC spec sheet][cat313] |
| Lower frame with rollers; swing frame; rest of the upper structure | 2,089; 989; 2,193 | The remaining 5,270 kg, split in the Cat 313 GC's proportions (2,620, 1,240 and 2,750 kg) |

Hitachi's own figures disagree by 147 kg. Its basic machine (9,500 kg) and front (2,220 kg) leave 280 kg for fuel, oils, coolant and operator. Item by item those come to 427 kg: fuel 205, hydraulic oil 113, engine oil 14, coolant 20 and operator 75, at assumed densities. The model keeps the 12,000 kg total.

Within each body the parts are boxes, rings, rods, points and thin shells at estimated positions:

- the upper structure: engine, pumps, tanks, valve, swing drive, batteries and cab;
- the lower: travel motors, idlers, seven rollers a side ([spec sheet][zx120]), carrier rollers, side frames and the two track loops;
- the boom, arm, bucket and covers: shells on their meshes.

The boom cylinder barrels (65/120 of the cylinder mass, assumed) ride on the upper structure and the rods on the boom, between the pins in `cylinders`. The bodies come to:

| Body | Mass, kg |
|---|---|
| Lower | 3,629 |
| Upper | 6,041 |
| Boom | 1,330 |
| Arm | 600 |
| Bucket | 400 |

Only the masses from the spec sheets are sourced. The positions of the other parts are estimates, and the lift chart below is the check on them.

**Track contact.** Each track meets the ground as a belt, not as points fixed to the frame. Every step, the height field is sampled:

- under the bottom of each track, at 121 stations 2.4 cm apart over the 2.88 m tumbler distance ([spec sheet][zx120]);
- at three points across each half of the 500 mm shoe;
- around each idler and sprocket, on an arc of radius 0.35 m (estimated from the track height).

Each half of a track is split into eight 0.36 m intervals. Each interval gives one contact at its deepest station. When stations tie within 1 mm, as on flat ground, a middle interval takes its middle station, and the front-most and rear-most touching intervals take their outer stations. Each idler arc gives one more contact, so a machine has at most 40.

An edge or step holds the track at the edge, to within one station. The contacts sit on the line where the rollers meet the track links, 7 cm either side of the track centre (assumed). That line is the side tipping line of [ISO 10567][iso10567]. The ground is rigid: at rest the soft contact sinks the machine by under 0.1 mm over 10 s, and adds no energy.

Friction differs along and across the track:

- **Along: 0.45 nominal.** This is the traction factor for crawler tracks on concrete in Caterpillar's handbook ([handbook][catphb]). It is unverified, because the table could not be read.
- **Across: 0.50 nominal.** This is just below the 0.55 to 0.58 for steel tracks on hard ground in a tracked-vehicles lecture based on Wong ([lecture][umd]).
- **Drawn ground:** 0.30 to 0.55 along, and 1.11 times that across (assumed).

MuJoCo's soft friction cone pushes a sliding body away from the surface: a box dragged across a plane reached 2 m/s upward, and counter-rotating tracks hopped. So a contact sliding faster than 5 cm/s is frictionless in MuJoCo's solver. For those contacts `tracks.step` applies Coulomb friction itself, elliptic between along and across, from the previous step's normal force. Slower contacts keep MuJoCo's friction, which gives static grip.

**Drive.** Each track has a belt driven through a travel valve built like the joint valves:

- **Commands:** the `travel_left` and `travel_right` actuators take a belt speed in m/s, up to 5.5 km/h (1.53 m/s).
- **Force cap:** 51 kN per belt, half the 102 kN maximum traction ([spec sheet][zx120]).
- **Inertia:** each belt carries half the track chain mass, 770 kg.
- **Parking brake:** with zero command the valve holds the belt, up to the same 51 kN. The spec sheet gives a spring-set travel brake but no capacity, so the cap is assumed.

A later hydraulics stage can drive the same two actuators and set the cap from travel pressure and motor size. Measured on flat ground at friction 0.45/0.50:

- **Straight**, forward and back: the commanded speed to within 1%.
- **Counter-rotation** at full speed: 0.88 rad/s, against 1.54 rad/s if neither track slipped. Each track pushes 17.8 to 18.5 kN. Wong's firm-ground turning resistance, μ_t W L / 4 over the 1.99 m gauge ([lecture][umd]), gives 21.3 kN at μ_t = 0.50. Wong puts all the friction across the track; the model splits it along and across.
- **Pivot turn**, one track braked and the other at full speed: 0.44 rad/s about a centre 1.86 m from the swing axis, outside the braked track. The braked track slides forward at about 0.38 m/s.
- **Drawbar pull** at friction 1.0: the machine keeps full speed against 90% of 102 kN and slows against 110%.
- **Gradeability** is set by friction, not traction. The machine slides down a slope steeper than the friction angle along the track: 24.2° at the nominal 0.45, tested at 0.95 and 1.05 times atan 0.45. The spec sheet's 35° would need a friction of 0.70. Drawn slopes stop at 20°.
- **Parked** at 0.45/0.50, the machine creeps:

| Slope | Creep, m/s |
|---|---|
| 10° | 1.3e-5 |
| 20° | 3.3e-5 |
| 23° | 3.6e-5 |
| 25.5° | slides |
| 15° across | 8e-6 |
| Flat | 3e-11 |

**Rough ground.** `draw_ground(seed)` draws ground for one machine, on its 13 by 12 m height field of 0.1 m cells:

- a plane tilted up to 20° in any direction;
- 0 to 2 steps of 5 to 40 cm;
- 0 to 2 kerbs, 10 to 20 cm high and 0.2 to 0.4 m wide;
- 0 to 2 ruts, 5 to 15 cm deep and 0.3 to 0.6 m wide;
- 0 to 6 smooth bumps or dips of up to 15 cm, with radii of 0.3 to 1.5 m;
- a friction pair.

The ranges are assumed for a 12 t machine on a site; none is sourced. Steps, kerbs and ruts run along the grid with their top edge on a grid line. The edge is therefore exact, and the height field's one-cell bevel falls on the low side.

The same seed gives the same ground. `apply_ground` writes a ground into one machine's height field and friction, and `place` sets a machine down at rest on whatever ground it has.

**Checks against hand statics.** Each check in `tests/test_machine_ground.py` brackets a hand calculation: below it nothing happens, above it the machine tips, slides or lifts. In the tipping checks the joints are held at a friction of 1e9 N·m, so these results do not depend on the joint force limits.

| Load at the bucket pin that tips the machine, with bucket | Hand statics, kg | Simulated, kg | Difference |
|---|---|---|---|
| Over the front, flat, 6.0 m out | 3,283 | 3,268 | −0.5% |
| Over the side, flat, 4.5 m out | 3,280 | 3,285 | +0.2% |
| Over the front, 10° nose down, 6.0 m out | 2,744 | 2,731 | −0.5% |
| Over the side, 10° cross slope, 4.5 m out | 2,553 | 2,541 | −0.5% |

The tests also bracket these, against the hand figure:

| Check | Bracket |
|---|---|
| Boom press-down force that lifts the front of the tracks | 0.9 and 1.1 |
| Slope where the machine slides | 0.95 and 1.05 times atan of the friction |
| Cross slope where the machine tips sideways | 0.95 and 1.05 |
| Tipping load with the front of the tracks over a 1 m drop, 1.2 m and 1.0 m ahead of the swing axis, tipping about the edge | 0.95 and 1.05 |

They also check that:

- a machine on drawn ground comes to rest;
- the same commands on the same ground repeat bit for bit;
- a welded machine gets no track contacts, and releasing it writes nothing into the model.

Against the [spec sheet][zx120]'s lift chart, the table gives 75% of the model's tipping load at every rating the chart marks as limited by tipping. It comes from `python -m hma.machine.statics`: no bucket, load at the bucket pin, kg.

| Height, radius (m) | Front: model / chart | Diff | Side: model / chart | Diff |
|---|---|---|---|---|
| 4.5, 6.0 | 3,086 / 3,040 | +1.5% | 2,155 / 2,110 | +2.1% |
| 3.0, 4.5 | | | 3,247 / 3,200 | +1.5% |
| 3.0, 6.0 | 3,014 / 2,980 | +1.1% | 2,089 / 2,050 | +1.9% |
| 1.5, 3.0 | | | 5,322 / 5,380 | −1.1% |
| 1.5, 4.5 | 4,474 / 4,470 | +0.1% | 2,985 / 2,980 | +0.2% |
| 1.5, 6.0 | 2,899 / 2,880 | +0.7% | 1,982 / 1,960 | +1.1% |
| 0.0, 3.0 | | | 4,954 / 5,070 | −2.3% |
| 0.0, 4.5 | 4,253 / 4,280 | −0.6% | 2,789 / 2,810 | −0.7% |
| 0.0, 6.0 | 2,799 / 2,790 | +0.3% | 1,891 / 1,880 | +0.6% |
| −1.5, 3.0 | 8,275 / 8,440 | −2.0% | 4,896 / 5,030 | −2.7% |
| −1.5, 4.5 | 4,162 / 4,200 | −0.9% | 2,707 / 2,740 | −1.2% |
| −1.5, 6.0 | 2,759 / 2,760 | 0.0% | 1,853 / 1,850 | +0.2% |
| −3.0, 3.0 | | | 4,990 / 5,120 | −2.5% |
| −3.0, 4.5 | 4,199 / 4,240 | −1.0% | 2,741 / 2,780 | −1.4% |

Every entry is within 2.7%. With the earlier mass split and sphere contacts, the model was 16 to 23% above the chart over the front and 24 to 31% over the side. The chart's hydraulically limited ratings, and how hard the boom can press down, do depend on the joint force limits, which are still constant caps.

**Left out**, with what each would change ([issue 29](../../issues/29)):

- **Track side faces.** Only the bottom of each track and its idler arcs touch the ground. A kerb or step face pushes on the idler arc only, so a track driven sideways into a kerb rides over it instead of being stopped.
- **Rollers, recoil springs and track sag.** The track is rigid between its idlers. On a bump shorter than the track it rests on the high points instead of bending around them between rollers. That puts the load spread on rough ground off; the tipping lines are not affected.
- **Travel hydraulics.** There is no travel circuit, crossover relief, two-speed shift or engine power limit. The travel force is the 102 kN cap at any speed; the real machine has less at high speed and more pull in low gear.
- **Rolling resistance.** None is modelled: on flat ground the tracks hold speed with about 0.02 N of drive force.
- **Brake creep.** The parking brake holds through a soft constraint, so a parked machine creeps by up to 3.6e-5 m/s on a slope, about 13 cm an hour. A spring-set brake holds still.
- **Other ground contact.** The belly, counterweight, boom, arm and bucket pass through the ground. A machine that rolls past about 85° falls through it.
- **Contact resolution.** An edge between two stations holds the track at the nearer inboard station, up to 2.4 cm in. With the drop exactly 1.0 m ahead, the tipping load was 3.0% under the hand figure. The height field's 0.1 m cells bevel every drop over one cell.
- **Speed.** A released machine steps about six times slower than a welded one, because the contacts are found in Python ([issue 28](../../issues/28)).

### Base controller

`BaseController` follows a bucket-tip path, given as position and pitch over time and turned into joint angles by inverse kinematics. Its command is the reference joint speed taken 0.17 s ahead, which covers the nominal machine's delay and lags, plus 0.25 /s times the joint angle error, divided by the nominal speed limit. It has no integral term. On the nominal machine it keeps the tip within 1 cm of a 3.1 m straight line driven in 4 s (mean 0.5 cm), at 10 Hz and at 20 Hz; with 0.5 s of delay the mean error grows to about 29 cm. The gain is set by the corner where every range is at its worst at once: 0.6 s delay, 0.4 s lag, 0.2 s sensor latency, curve gain 1.1, speed limits 1.2 times nominal, the steepest allowed first curve segment, full joint play and sensor noise. That machine moves up to 2.6 times faster per unit of command than the nominal one and responds about 1.25 s late, which puts the stability limit near 0.4 /s; 0.25 /s leaves a margin of about 1.6. At 1 /s the corner machine still swings by 1 to 4 m 8–10 s after the line ends, and 10 of 300 randomly drawn machines swing by more than 10 cm. At 0.25 /s no drawn machine does, and the corner machine without dead zones settles to within 1.4 cm in 30 s. The price is a large steady offset, because commands smaller than a joint's dead zone do nothing and there is no integral term to grow them. Over 300 randomly drawn machines at 10 Hz, 10 s after the line ends, the tip sits a median 133 cm and at worst 287 cm from its target. With the dead zones set to zero on the same machines, the median falls to 4 cm (at worst 66 cm), and the machines still moving keep converging without swinging.

[nh]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/cb538c33-9f47-4539-b238-2ef6372e62d2/content
[eh]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/b00c6c58-12a1-49af-a587-16872fe608a9/content
[e22]: https://www.research-collection.ethz.ch/server/api/core/bitstreams/95ef5691-11e8-4a86-b02d-6f0e2501de9b/content
[heap]: https://arxiv.org/abs/2106.05059
[sp]: https://arxiv.org/abs/2410.05093
[zx120]: https://machineryline.kr/img/pdf/7/3/1688885834710464825/KA-EN200ID_ZX120.pdf
[zxman]: https://youfixthis.com/d/hitachi-zaxis-110-110m-120-130-130lcn-125us-135us-135ur-repair-manual-excavator
[iso6015]: https://www.iso.org/standard/12217.html
[zx5g]: https://www.hitachicm.com/ae/en/machinery/excavators/medium-excavators/product.zx130-5g
[zx5b]: https://www.hitachicm.com/us/en/products/excavators/medium-large/product.zx130-5b-la
[cat313]: https://bandtplanthire.co.uk/wp-content/uploads/2023/07/313GC-TECH.pdf
[iso10567]: https://cdn.standards.iteh.ai/samples/18652/9506b3e367af44b68c0b902437c1eda8/ISO-10567-1992.pdf
[catphb]: https://www.ringpower.com/media/2086/caterpillarperfhandbook_ed50.pdf
[umd]: https://spacecraft.ssl.umd.edu/academics/788XF24/788XF24L10.tracks.pdf

## Feature map

A feature is **done** when it runs from a clean checkout and its checks pass.

| Feature | What it does | Status |
|---|---|---|
| Machine model | A 12 t class excavator with swing, boom, arm and bucket joints. Each joint takes a speed command and has a force limit. The base can be held still, or left free on two driven tracks to travel, turn, slide and tip | Done |
| Hydraulics model | Gives each machine its own delay, sluggishness, dead zone, lever-to-speed curve, flow limit, speed and force limits, slack in the joints, weights, and sensor noise and latency | Done |
| Baseline controller | A conventional controller tuned once on the standard machine | Done: follows paths in free air |
| Rough ground | Slopes, steps and edges of rigid ground. The machine can slide, rock and tip, and the arm and bucket meet the ground | In progress: ground drawn from a seed, and the tracks stand, drive, slide and tip on it, checked against hand statics; the arm and bucket do not touch it yet |
| Carried load | A load with real mass on the bucket | Planned |
| Planner | Turns a checked specification into paths for the bucket and the load | Planned |
| Cylinder forces | Joint force limits from the hydraulic cylinders' sizes and positions | In progress: geometry, sizes and static checks done; the simulation still uses constant caps |
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

The first command opens a viewer and moves the arm through its range. The second asks a GPT model for a trench plan, the task type of the language step until it is changed, and checks it. It calls the [Codex CLI](https://github.com/openai/codex), which must be installed and signed in with your own ChatGPT account; this repo holds no API key.

## Limits

- Simulation only. Nothing here has been checked against a real excavator.
- Only the tracks touch the ground. The arm and bucket pass through it without force.
- The track model leaves out side faces, rollers and track sag, travel hydraulics and rolling resistance, and its parking brake creeps ([issue 29](../../issues/29)).
- A released machine steps about six times slower than a welded one ([issue 28](../../issues/28)).
- Calibrating the controller to a specific real machine is future work.

## Licence

The code is under the [PolyForm Noncommercial License 1.0.0](LICENSE.md): you may read, run and change it for noncommercial purposes. Third-party files keep their own licences and are credited where they are added. The machine model is converted from [pwri-opera/zx120_ros](https://github.com/pwri-opera/zx120_ros) under Apache-2.0; its [notice](src/hma/machine/assets/NOTICE) lists the changes.

Required Notice: Copyright 2026 Gabriel Ezerzer
