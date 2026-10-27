# The oracle writer in detail: a walkthrough of the code

This page follows one run of

```
python -m vla_memory.proxy --writer oracle --method subgoal --task vla_memory/tasks/red_blue_blocks.yaml \
    --listen-port 8001 --upstream-port 8000
```

from the moment you press Enter to the last policy call, and names the function that does each step. For the reference
task you can leave out `--task`. The short page, [docs/writers/oracle](../README.md), says what the oracle is for. The
start-up (section 1) and the handling of a policy call (section 3) are the same for every writer, so the other writers'
pages point here. Section 9 lists the steps to use the oracle with your own simulator.

![One policy call with the oracle writer and the sub-goal method](oracle_call.svg)

## The pieces

| file | what it does in this run |
|---|---|
| `vla_memory/proxy/__main__.py`, `proxy/cli.py` | reads the command line, builds everything, starts the server |
| `vla_memory/task.py` | `TaskSpec`: reads the task file, the skills and their order |
| `vla_memory/writers/base.py` | what every writer can do (`Writer`), what a writer sees of a call (`Call`), the list of writers (`WRITERS`) |
| `vla_memory/writers/oracle.py` | the oracle writer (`OracleWriter`) and the writer `off` (`OffWriter`) |
| `vla_memory/methods/base.py`, `methods/subgoal.py` | what every method can do (`Method`), and the sub-goal method (`SubgoalMethod`) |
| `vla_memory/proxy/server.py` | `ProxyServer`: the websocket server the environment connects to |
| `vla_memory/proxy/relay.py` | `Relay`: what happens at every policy call |
| `vla_memory/transport.py` | `OpenPIWebsocketTransport`: the connection to the policy server |
| `vla_memory/env_keys.py` | `ControlKeys`: a helper for the environment that builds the `memory/` entries it sends |

The simulator is not part of the framework, and neither is its **tracker**, the part of the simulator that knows the
current skill. The framework only defines what the simulator has to send (section 2).

## 1. Start-up: from the command line to a waiting proxy

`python -m vla_memory.proxy` runs `vla_memory/proxy/__main__.py`, which calls `cli.main()`. Then, in this order:

1. **The command line is read in two passes** (`parse_command_line`). The first pass (`parse_choices`) only looks for
   `--writer`, `--method` and `--with`, so that the program knows which parts it needs. `resolve_writer("oracle")` finds
   the name in the list `WRITERS` (`writers/base.py`) and only now imports `vla_memory.writers.oracle`: a writer you did
   not choose is never loaded. The method `subgoal` is found the same way in `METHODS` (`methods/base.py`). The second
   pass reads everything, including the options the chosen parts add themselves (`add_arguments`). The oracle adds none.
2. **The options are checked** (`check_arguments`). The default connection to the policy needs `--upstream-port`.
3. **The task file is loaded** (`TaskSpec.load_or_default`). The oracle only needs the skills and their order:
   - the objects (`red`, `blue`) and the verbs (`pick up`, `place`);
   - the skill template `"{verb} the {object} block"`, which turns a verb and an object into the exact text of a skill,
     such as "pick up the red block" (`TaskSpec.skill`);
   - the task order (`TaskSpec._task_order`): stage by stage, and inside a stage one object after the other. The reference
     task has a single stage, so the order is: pick up red, place red, pick up blue, place blue.
4. **The parts are built** (`build_components`):
   - the extensions (none in this run);
   - the writer, `OracleWriter(task)`. It prepares a fresh episode right away (`start_episode`):
     - `plan`: the task order, `["pick up the red block", "place the red block", "pick up the blue block", "place the blue
       block"]`;
     - `skill_index = 0`: the position of the current skill in the plan, here the first one;
     - `passthrough = None` (section 3, step 5) and `reported = None` (section 5);
     - `done_all = False`, meaning "the whole plan is finished". The oracle never sets it to True: its tracker only reports
       the current skill, so the last skill stays current until the episode ends.
   - the method, `SubgoalMethod(task)`;
   - each extension is told the writer's `monitor`, the part of the writer that watches the robot. The oracle watches
     nothing itself, so its `monitor` is `None`.
5. **The connection to the policy is opened** (`build_transport`). `OpenPIWebsocketTransport` uses openpi's client
   library, which connects to the policy server and waits until it answers. The `Relay`, the part that handles every
   call, is built around this connection and writes the policy server's self-description (its metadata) to the log. With
   `--record-head DIR` a recorder for the [head writer](../../head/README.md) is built too (`build_recorder`).
6. **The proxy starts listening** (`serve`). `ProxyServer.run` opens a websocket server on the listen port (on 127.0.0.1
   unless `--listen-host` says otherwise) and logs `proxy listening on 127.0.0.1:8001 (writer oracle, method subgoal)`. A
   request to `/healthz` answers `OK`, so launch scripts can wait until the proxy is up.

When the environment connects, the proxy first tells it what it is (`proxy_metadata`):
`{"proxy": "vla_memory.proxy.relay", "mode": "oracle", "method": "subgoal", "task": "red_blue_blocks"}`. A writer with
settings of its own adds them; the VLM writer, for example, adds its model, its example store and `k`.

## 2. The simulator's side: what it sends, and how it knows the current skill

The simulator connects to the proxy the same way it would connect to the policy server, receives the description above,
and then sends one observation per policy call. It decides itself how often to call; the reference setup calls every 8
control steps and carries out the returned actions in between. An observation contains:

- the policy's usual entries, unchanged: in the reference setup the two camera images (`observation/image`,
  `observation/wrist_image`) and the robot's state (`observation/state`);
- `prompt`: the full task instruction, for example "Pick up the red block, place it back, and then grasp the blue block.";
- entries for the proxy, whose names start with `memory/`. The oracle reads:
  - `memory/oracle_prompt`: the skill the simulator considers current, in exactly the words of the task file;
  - `memory/new_episode` (True at the first call of an episode) and `memory/episode` (the episode's name), so that the
    proxy knows where an episode begins;
  - `memory/objects`, but only when an episode holds just some of the task's objects (section 5);
  - `memory/step`, which goes into the log.

  Other `memory/` entries, such as the gripper readings or the raw camera pictures, are for the other writers and for some
  methods (the keyframe method takes its picture from `memory/wrist_raw`). The oracle itself ignores them.

`vla_memory/env_keys.py` builds these entries: `ControlKeys.start_episode(episode_id, objects)` at the start of an
episode, `record(...)` at every control step (for the writers that need the readings of every step), and
`call(step, oracle_prompt=...)` at every policy call, which returns the entries to add to the observation.
`observation(...)` does the last step and adds the policy's entries and the prompt too (section 9).

**How the simulator knows the current skill.** The simulator knows the true state of its world: where each cube is, what
the gripper was told to do, where the fingertips are. The reference simulator turns this into the current skill with four
yes/no flags. They are set in the task's order and are never cleared during an episode:

| flag | set when |
|---|---|
| red is lifted | the red cube's centre is more than 9 cm above where it rests (above 0.85 m; the table top is at 0.74 m, a resting cube's centre at 0.76 m) |
| red is placed | red was lifted, the gripper was told to open, the cube rests on the table again (within 1 cm of its resting height), and the fingertips have moved at least 6 cm above it or 8 cm away from it |
| blue is lifted | red is placed, and blue's centre is more than 9 cm above where it rests |
| blue is placed | as for red |

The current skill is the first one whose flag is not set yet: "pick up the red block" until red is lifted, then "place the
red block" until red is placed, then "pick up the blue block", then "place the blue block", which stays current until the
episode ends.

Why "placed" waits until the fingertips have moved away: the policy learned from demonstrations whose labels switch from
"place" to the next skill only after the hand has pulled back from the cube. When the tracker switched as soon as the
gripper opened, the policy got the next prompt while its fingers were still around the cube, and it grabbed the same cube
again. The tracker has to switch where the training labels switch.

## 3. One policy call, step by step

`ProxyServer._handle_connection` unpacks every message into a Python dictionary and passes it to `Relay.infer`, in a
separate thread so that the server stays responsive. `Relay.infer` then works through these steps:

1. **Sort the entries** (`split_control_keys`). Every `memory/` entry goes into `control_keys`, without the prefix
   (`memory/oracle_prompt` becomes `oracle_prompt`). Everything else stays in the observation for the policy. The
   instruction is the observation's `prompt`.
2. **Start the record of this call**: `info = {"mode": "oracle", "queried": False, "advanced": False, "vlm_ms": 0.0}`.
   The oracle never changes the last three; they are there for the other writers.
3. **Check for a new episode** (`_is_new_episode`): the very first call, `memory/new_episode` set to True, or a new
   `memory/episode` name. If so, `_begin_episode`:
   - closes the previous episode (`_end_episode`: the extensions and the recorder take it; there are none here);
   - names the new one: the `memory/episode` value, otherwise `ep0`, `ep1`, ...;
   - tells the extensions and the method (the sub-goal method has nothing to do here);
   - tells the writer (`writer.start_episode`), with the objects if the simulator sent `memory/objects`. The oracle goes
     back to the first skill of a fresh plan.
4. **Package what the writer may look at**: `Call(instruction, control_keys, observation, image_keys)`.
5. **Let the writer decide, before the policy is called** (`OracleWriter.before_call`):
   - It takes the text of `memory/oracle_prompt`. If that entry is missing, it takes the instruction instead and writes
     one warning to the log for the whole run.
   - If the text is one of the plan's skills, that skill becomes the current one (`_point_at` sets
     `skill_index = plan.index(...)`). The oracle trusts the simulator completely and goes wherever it is told.
   - If the text is not a skill, the oracle stores it as `passthrough`, and the current skill stays where it was. This
     happens, for example, when a simulator without a tracker sends the full instruction.
6. **Let the method set the prompt** (`_decide_prompt`). `SubgoalMethod.prompt` asks the writer for its current skill
   (`writer.current_skill`): the passthrough text if there is one, otherwise `plan[skill_index]`. This replaces the
   observation's `prompt`, and the record notes `prompt` and `oracle_prompt`. Then the method and the extensions may add
   entries to the observation (`before_policy`). The other methods do this instead:
   - `--method history`: the prompt becomes the instruction plus one line per finished skill (or the passthrough text, if
     there is one). "Finished" means every skill before the current one in the plan (`Writer.completed`, which returns
     `plan[:skill_index]`); the history shows at most the task's `history.max_lines` of them, and never the plan's last
     skill.
   - `--method keyframe`: the prompt stays the instruction, and `before_policy` adds a third picture
     (`observation/memory_image`): the wrist picture from the call at which the last skill was finished. Before the first
     skill is finished there is none.
7. **Put the episode markers back** (`episode_markers`). `memory/new_episode` and `memory/episode` go to the policy as the
   simulator sent them, so that a policy with a memory of its own can reset it. All other `memory/` entries stay in the
   proxy.
8. **Call the policy** (`_forward`). `transport.infer` sends the observation to the policy server and waits for the
   answer: a dictionary with the `actions`. The time this takes is noted as `upstream_ms`. With `--record-head DIR`, the
   recorder now keeps the policy's state from the answer, for fitting the head writer.
9. **Let the extensions look at the answer** (`_reply_hooks`); there are none here.
10. **Give the writer a second chance, after the call** (`writer.after_call`). The oracle has already decided and does
    nothing; the head writer decides here.
11. **Finish the record.** The total time is noted (`total_ms`), the record is attached to the answer as `planner`, and
    with `--log FILE` it is written as one JSON line, for example:
    `{"episode": "seed1001", "step": 96, "mode": "oracle", "queried": false, "advanced": false, "vlm_ms": 0.0,
    "prompt": "place the red block", "oracle_prompt": "place the red block", "upstream_ms": ..., "total_ms": ...}`.
12. **Send the answer back.** The simulator receives the actions plus `planner` and carries out the actions until its
    next call. It may ignore `planner`; the reference simulator stores it, so that its records show which prompt the
    policy got at every call.

Because the oracle decides in step 5, before the policy is called, a new skill reaches the policy in the same call in
which the simulator reports it. There is no delay of one call.

## 4. A whole episode of the reference task

| calls | `memory/oracle_prompt` | what the oracle does | the policy's prompt (sub-goal) |
|---|---|---|---|
| the first call | "pick up the red block", with `memory/new_episode` True | starts a new episode, `skill_index = 0` | "pick up the red block" |
| until red is 9 cm up | "pick up the red block" | stays at `skill_index = 0` | "pick up the red block" |
| from the first call after the lift | "place the red block" | `skill_index = 1` | "place the red block" |
| from the first call after the hand has left the red cube | "pick up the blue block" | `skill_index = 2` | "pick up the blue block" |
| from the first call after blue is lifted, to the end | "place the blue block" | `skill_index = 3` | "place the blue block" |

The next episode begins when the simulator sends `memory/new_episode` set to True (or a new `memory/episode` name), and
the oracle starts again at the first skill. Whether the task actually succeeded is not the writer's business: the
simulator's own success measure decides that.

## 5. Episodes with varying objects

Some tasks put only some of their objects into an episode, say two to four cubes out of five colours. The simulator then
names the episode's objects in `memory/objects`, and the relay hands them to `start_episode`:

- `plan = task.default_plan(objects)`: the task order, but only for these objects;
- `reported = []`: an empty list of the skills reported so far.

From then on, every skill the simulator reports for the first time is added to `reported` (`_point_at`), and the plan is
rebuilt as "the skills reported so far, then the rest in task order". So the plan follows the order in which the simulator
reports the skills, and "finished" (the skills before the current one) stays right even when the task allows any order.
An example with three cubes, where the simulator reports green before red:

| reported so far | plan | current skill |
|---|---|---|
| nothing | the task order of red, green and purple | the first skill of that order |
| "put the green cube into the box" | that skill first, then the rest in task order | "put the green cube into the box" |
| "put the green cube into the box", then "put the red cube into the box" | those two first, then the rest | "put the red cube into the box"; the green one counts as finished |

## 6. What the oracle does not do

- It has no model, looks at no camera and uses no threshold of its own. Everything it knows comes from
  `memory/oracle_prompt`.
- It is not a monitor (`monitor` is `None`). Extensions that work through a monitor, such as the session store, which
  belongs to the VLM writer, have nothing to work with.
- It never marks the whole plan as finished (`done_all` stays False).
- It needs a tracker, so it works only in simulation. On a real robot, the rules writer, the VLM writer or the head writer
  takes its place.

## 7. The writer `off`, in the same file

`--writer off` is `OffWriter`: an empty plan, no current skill, nothing ever finished. It also tells the relay not to touch
the prompt (`drives_prompt = False`), so in step 6 the policy gets the simulator's instruction unchanged. It is the run
without any writer: the memoryless baseline.

## 8. Stopping and checking a run

**Stopping.** Ctrl-C or SIGTERM (as launch scripts send it) tells the server to stop (`serve_until_stopped`). The server
accepts no new connection, closes the open ones and waits until the running call has finished. Then `Relay.close` ends
the last episode for the extensions and the recorder, so that they save it, and closes the log.

**Checking.** With `--log FILE`, every line of the log is one call. In an oracle run:

- `prompt` equals `oracle_prompt` in every call, unless the simulator sent a text that is not a skill;
- a skill switches where `oracle_prompt` changes from one line to the next;
- `upstream_ms` is the time the policy took for the call, and `total_ms` the time of the whole call through the proxy.

## 9. Using the oracle with your own simulator, step by step

Nothing in the framework changes; the work is on the simulator's side.

1. **The task file.** Write it ([Creating a new task](../../../tasks/README.md)) and check that its skills are exactly
   the prompts the policy was trained on:

   ```
   python -c "from vla_memory import TaskSpec; print(*TaskSpec.load_or_default('task.yaml').skills, sep='\n')"
   ```

2. **The tracker.** Make your simulator work out the current skill at every policy call from its true state, switching
   where the policy's training labels switch (section 2). It must use exactly the words of the task file;
   `TaskSpec.skill(verb, object)` gives them, for example `task.skill("place", "red")`.
3. **The entries.** Send the current skill as `memory/oracle_prompt`, and mark the start of every episode with
   `memory/new_episode` (and, if you like, a name in `memory/episode`). For episodes with only some of the task's objects,
   also send `memory/objects`. `ControlKeys` does all of this:

   ```python
   from openpi_client import websocket_client_policy
   from vla_memory import ControlKeys, TaskSpec

   task = TaskSpec.load_or_default("task.yaml")
   proxy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=8001)   # the proxy, not the policy
   keys = ControlKeys(task)
   keys.start_episode("seed1001")                 # objects=[...] for an episode with only some of the objects
   for step in range(max_steps):
       if step % 8 == 0:                          # a policy call every 8 control steps
           observation = keys.observation(policy_keys, instruction, step, oracle_prompt=current_skill)
           actions = proxy.infer(observation)["actions"]
       ...                                        # carry out the actions until the next call
   ```

   `policy_keys` holds the policy's usual entries (the pictures and the state), `instruction` is the full task
   instruction, and `current_skill` is your tracker's answer. For the keyframe method, also pass the raw wrist picture
   (`wrist_raw=...`).
4. **Start the policy server**, then the proxy with a log:

   ```
   python -m vla_memory.proxy --writer oracle --method subgoal --task task.yaml --log oracle_run.jsonl \
       --listen-port 8001 --upstream-port 8000
   ```

   The proxy logs the policy server's metadata (`upstream metadata: ...`) and then
   `proxy listening on 127.0.0.1:8001 (writer oracle, method subgoal)`.
5. **Run the simulator** and check the log as in section 8: `prompt` should follow `oracle_prompt` from call to call. The
   warning `oracle writer without memory/oracle_prompt` means that the simulator sends no current skill.
