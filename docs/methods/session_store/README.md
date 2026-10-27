# The session store: the long-term memory on a vector database

## The idea

**The session store is not a writer of its own.** It is a part of the [VLM writer](../../writers/vlm/README.md): the writer's
long-term memory of past moments, in place of the example store of `--memory`. The VLM writer still decides when a skill is
done. The session store changes where that memory is kept, and lets it grow from the robot's own episodes.

The VLM writer decides at every policy call whether the current skill is done, and it asks a large model to find out. That is
slow: a question takes the model about 0.7 s, and 2.9 s when four example pairs are attached, while a policy call takes only
0.17 s. A memory of past moments can answer many of these questions by itself.

**The memory.** It is a store of **moments** of past episodes. A moment is the two camera pictures of one policy call and
what was true then: which skill was active, where each object was, whether the gripper was closed. At a call, the 6 stored
moments that look most like the current pictures **vote**: if none of them says "done", the model is not asked at all. The
4 nearest are also shown to the model as examples, each with its right answer.

**The key.** To find the moments that look most like the current pictures among thousands, the store cannot compare the
pictures pixel by pixel: that is slow, and a little change of light or position changes many pixels. So every moment gets a
**key**, a short list of numbers that sums up its pictures. Two moments count as alike when their keys point the same way,
and a **vector database** finds the nearest keys in milliseconds. There are two kinds:

- **The colour key** (the default) is made by hand: a 12 × 16 colour thumbnail of each picture plus, for each object colour
  of the task, how many pixels have it, where they are and how spread out. It says directly where the red and the blue
  block are and whether one is in the gripper. It needs no model and takes 3 ms, but it only works with objects of clearly
  different colours, which the task file names.
- **A learned image key** comes from a network trained on millions of pictures (DINOv2 or SigLIP). It turns the pictures
  into 384 numbers (DINOv2, the exterior picture) or 1,536 (SigLIP, both pictures) that describe what they show in
  general. It needs no colours, so it could serve other scenes, but it also reacts to details that do not matter for the
  task, and it takes 58 to 207 ms on the CPU.

**Before the session store**, the memory was the **example store**: one array file (`--memory`), built once from recorded
demonstrations and never changed. The planner reads the session store in exactly the same way: the same vote and the same
examples, written by the same code. What changes is where the moments come from and how they are kept:

| | the example store (`--memory`) | the session store (`--with session_store`) |
|---|---|---|
| the moments come from | recorded demonstrations, put in once by a script | the robot's own episodes, added after every episode; it can start from imported demonstrations |
| labelled by | the demonstrations' own phases | the rules writer, replayed afterwards over the robot's gripper signals |
| during a run | never changes | grows after every episode |
| kept in | one array file | a database directory (LanceDB: a directory of files, no server) |
| the key | the colour key | the colour key, or a learned image key |
| a lookup can be limited to | nothing: every moment counts | the same task, instruction, plan or session |
| each moment also records | nothing more | where it came from, who labelled it, how often it was retrieved |
| size | fixed | unbounded, or capped with a rule for what goes |

![The session store: what happens at every call, and when an episode ends](detailed/session_store_overview.svg)

**What it gives over the example store:**

- **It needs no demonstrations.** It can start empty and fill itself from the robot's own episodes, on any task the rules
  writer can label.
- **It holds the robot's own situations,** slow and failed grasps included. A failed episode with right labels gives right
  "not done" examples; demonstrations show only clean runs.
- **It answers more of the checks by itself as it grows:** about a fifth in the first ten episodes, a third after about
  twenty (findings below).
- **It stays manageable.** Every moment records where it came from and who labelled it, so the store can be checked and
  pruned. A lookup can be limited to the same task, instruction or session, so one store can serve many. A cap with a rule
  for what goes keeps it from growing without end, and a new episode is appended without rewriting a file.

**Where the labels come from.** A moment can only vote if it is labelled: which skill was active, and what was true then.
During an episode the store keeps the pictures and the gripper signals of every call. When the episode ends, the calls are
labelled in hindsight by the **rules writer**, replayed over the robot's own gripper signals. The labeller cannot be the
planner itself: the store is there to check the planner, and every early "done" of the planner would be stored as a false
"done" moment that then votes for more of them. The simulator's ground truth exists only in simulation. The rules writer
needs neither a simulator nor a model, and it switches in the same policy call as the ground truth in 275 of 286 cases
([its page](../../writers/rules/README.md)). Calls after the task was complete are left out, and so are near-duplicates: an
episode keeps at most two calls with the same labels and almost the same colour key. The colour key decides this whatever
key retrieves, because it is cheap and always stored, so every store keeps the same calls.

**What it cannot do:**

- **It did not raise the task success, and it did not make the episodes shorter.** The five live runs completed 88 to 94
  of 100 episodes; the VLM writer with no store completed 94 in an earlier run on other seeds. The store saves questions to
  the planning model, but the questions that remain carry four examples each and take longer.
- **The colour key needs clearly coloured objects.** A learned key needs none, but it was not tried in a live run.
- **Search gets slow in a big store** (about 250 episodes give 10,000 moments). Beyond that the store needs an index or a cap.
- **The labels are the rules writer's.** A moment is as right as the rules writer's reading of the robot's gripper signals. A
  task without grasps needs rules of the `signals` kind first. Moments are labelled in the task's order of skills, so an
  episode that handles the objects in another order cannot be stored faithfully. A wrong label stays until the moment is
  evicted.
- **The filters were not needed here.** Unrelated sessions did not hurt the monitor in our test, so the filters are untested
  where they would matter: where the same picture means a different answer under another instruction.
- **It needs more packages:** `lancedb` and `pyarrow` for the database, and `torch` and `transformers` for the learned keys.

How it works in the code, from the command line to the planner's questions and to the write at the end of an episode:
[detailed/README.md](detailed/README.md).

## Prerequisites

- The VLM writer with the planning model (see the VLM writer's page), and the environment's gripper signals
  (`memory/proprio_steps`, or the current finger values) for the write path, as for the rules writer.
- Python packages: `lancedb` and `pyarrow` for the database (`pip install -e '.[session_store]'`). The keys `dinov2` and
  `siglip` also need `torch` and `transformers` (`pip install -e '.[session_store_encoders]'`). The default colour key needs
  neither.
- For `dinov2` and `siglip`, the encoder weights, downloaded once into the Hugging Face cache: `facebook/dinov2-small` and
  `google/siglip-base-patch16-224`. They run on the CPU (58 ms for a DINOv2 key, 207 ms for a SigLIP key) or on a GPU.
- Optional: an example store file (`scripts/build_exemplar_store.py`) to import as demonstrations. Otherwise the store starts
  empty and grows from the robot's own episodes.

## How to use it in the framework

The store is an extension of the proxy. It gives the planner its memory (in place of `--memory`), records the calls and writes
the finished episodes. `--task task.yaml` names the task file; for the reference task it can be left out.

```
python -m vla_memory.proxy --writer vlm --method subgoal --task task.yaml --listen-port <port> --upstream-port <policy port> \
    --vlm-url http://127.0.0.1:8100/v1 --log run/proxy.jsonl \
    --with session_store --store-dir stores/session_a --store-filter instruction --store-session reference
```

The colour key retrieves unless `--store-key dinov2` or `siglip` says otherwise. A new database stores the colour key, which
also decides the near-duplicates of an episode, and the retrieval key; `--store-keys` adds more (`--store-keys dinov2 siglip`
for all three). An existing database keeps the keys it was created with and retrieves with any of them; asking for another is
a usage error. Every new moment gets all of the database's keys, so a database with a learned key needs `torch` and
`transformers` to grow, even when the colour key retrieves.

| option | meaning | default |
|---|---|---|
| `--store-dir DIR` | the database directory; created when missing, resumed when it exists | required |
| `--store-import FILE` | an example store (`.npz`) imported as demonstrations when the database is new | none |
| `--store-import-instruction TEXT` | the instruction recorded with the imported demonstrations | the task file's first instruction example |
| `--store-key colour\|dinov2\|siglip` | the key that retrieves | colour |
| `--store-keys KEY [KEY ...]` | further keys a new database stores with every moment, besides the colour key and the retrieval key | none |
| `--store-filter none\|task\|instruction\|plan\|session` | retrieve only moments with the same task, instruction text, plan or session | none |
| `--store-session NAME` | the session written with every new moment (and matched by the session filter) | the start time |
| `--store-write` / `--no-store-write` | add the robot's own episodes after each episode | on |
| `--store-labeller rules\|oracle` | who labels the robot's calls: the rules writer on the robot's signals, or the simulator's prompt | rules |
| `--store-cap N` | most moments kept; 0 means unbounded | 0 |
| `--store-evict fifo\|use_recency` | which moments go above the cap | use_recency |
| `--store-device cpu\|cuda` | where the encoders of the learned keys run | cpu |
| `--store-threads N` | CPU threads of the encoders | 8 |

The vote and the examples keep their own options (`--k`, `--k-vote`, `--gate-min-done`). With `--log FILE`, the planner's own
log (`FILE.planner.jsonl`) notes the retrieval time and the store size in its monitor records, and writes one `store_update`
record per episode: what was added, what was evicted, the time spent and the size on disk. A store directory can also be read
outside the proxy with `vla_memory.methods.session_store.store.SessionStore` (open, query by key and filter, insert, evict),
for example to re-encode an existing store with another key and measure how well it retrieves.

## Our findings

Five live runs with the sub-goal policy on the reference task (two blocks, picked up and put back in order) and the VLM
writer with the vote and the colour key. Each run had 100 episodes of 400 control steps and a store that started empty. In
two runs, two other sessions wrote into the same store as well (20 episodes each with the instructions "only the red
block" and "the blue block first"), once without and once with the filter "same instruction". Three runs tested a size cap
of 1,000 moments: the least used go first, the oldest go first, or no cap. Per run we counted:

- **Full tasks:** both blocks picked up and put back.
- **Questions per episode:** how often the planner asked the planning model.
- **"Done" precision:** of the planner's "done" verdicts, the share that were right.

| run | full tasks | questions per episode | "done" precision | moments at the end | median episode |
|---|---|---|---|---|---|
| other sessions write in, no filter | 90 of 100 | 27.7 | 0.99 | 5,960 | 172 s |
| other sessions write in, filter "same instruction" | 88 of 100 | 28.6 | 0.98 | 5,881 | 182 s |
| cap of 1,000, the least used go first | 91 of 100 | 28.5 | 0.97 | 1,000 | 180 s |
| cap of 1,000, the oldest go first | 94 of 100 | 28.8 | 0.95 | 1,000 | 183 s |
| no cap | 91 of 100 | 27.7 | 0.97 | 3,983 | 179 s |

- **The store takes over part of the model's work.** In the first ten episodes the vote settled 17 to 23 % of the
  planner's checks without asking the model; after about twenty episodes, 31 to 42 %. The questions per episode fell from
  32 to 36 to 25 to 30, and no episode ended on a false "done".
- **Other sessions did not mislead it.** In the run without a filter, 28 % of the moments it retrieved came from the other
  sessions, and its "done" precision stayed at 0.99 (0.98 with the filter). Those moments show the same skills with their
  right outcome. A filter would matter where the same picture means a different answer under another instruction.
- **A cap of 1,000 moments (about 24 episodes) cost no full tasks.** With the oldest going first, the planner asked more
  questions once the cap was reached than without a cap (more in 47 episodes, fewer in 22, seed by seed); with the least
  used going first it did not (41 and 30).
- **Time.** Episodes took a median of about 3 minutes and did not get shorter as the store grew: the vote skips questions,
  but the questions that remain carry four examples (2.9 s per question instead of 0.7 s). Up to two runs at a time shared
  one planning model. Writing an episode into the store took a median 10.7 s, at the start of the next episode, nearly all
  of it computing the learned keys, which these runs stored as well.
- **A bigger store.** A search takes 10 ms at 1,000 moments, 25 ms at 10,000 and 212 ms at 100,000 (exact search on the
  CPU, in stores grown to that size from recorded moments); an index on the DINOv2 key keeps it at 15 to 17 ms. A query's
  key takes 3 ms for the colour key, 58 ms for DINOv2 and 207 ms for SigLIP. Inserting an episode takes 15 ms, and a moment
  takes about 0.12 MB on disk.
