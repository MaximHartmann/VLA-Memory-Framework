# The session store: the long-term memory on a vector database

## The idea

**The session store is not a writer of its own.** It is a part of the [VLM writer](../../writers/vlm/README.md): the writer's
long-term memory of past moments, in place of the example store of `--memory`. The VLM writer still decides when a skill is
done. The session store changes where that memory is kept, and lets it grow from the robot's own episodes.

The VLM writer decides at every policy call whether the current skill is done, and it asks a large model to find out. That is
slow: a question takes the model about 0.7 s, and 2.9 s when four example pairs are attached, while a policy call takes only
0.17 s. A memory of past moments can answer many of these questions by itself.

The writer's long-term memory is a store of **moments** of past episodes. A moment is the two camera pictures of one policy
call and what was true then: which skill was active, where each object was, whether the gripper was closed. At a call, the
stored moments that look most like the current pictures are retrieved. They **vote** on the question: if none of them says
"done", the model is not asked at all. The nearest of them are also shown to the model as examples, each with its right
answer. "Looks most like" is measured with a **key**, a list of numbers per moment: by default a hand-made one, a coarse colour
thumbnail of each picture plus the area and position of each object colour.

**Before the session store**, this memory was the **example store**: one array file (`--memory`), built once from recorded
demonstrations and never changed. The session store takes its place. The planner reads it in exactly the same way: the 6
nearest moments vote, the 4 nearest become the examples, and the same code writes them. What changes is where the moments
come from and how they are kept:

| | the example store (`--memory`) | the session store (`--with session_store`) |
|---|---|---|
| read by the planner for | the vote and the examples | the same, in the same way |
| kept in | one array file | a database directory (LanceDB: a directory of files, no server) |
| the moments come from | recorded demonstrations, put in once by a script | the robot's own episodes, added after every episode; it can start from imported demonstrations |
| labelled by | the demonstrations' own phases | the rules writer, replayed afterwards over the robot's gripper signals |
| during a run | never changes | grows after every episode |
| "looks most like" is measured by | the colour key | the colour key, or a learned image key |
| a lookup can be limited to | nothing: every moment counts | the same task, instruction, plan or session |
| each moment also records | nothing more | where it came from, who labelled it, how often it was retrieved |
| size | fixed | unbounded, or capped with a rule for what goes |

So the session store is the same memory in a database (a **vector database**, which finds the rows whose key is nearest to a
query key), plus a **write path** that fills it from the robot's own experience. One row of the database is one moment:

```
keys         the colour key (always), and optionally a DINOv2 ViT-S/14 embedding of the exterior picture
             and a SigLIP embedding of the exterior and wrist pictures
pictures     the exterior and the wrist picture (PNG)
payload      session, task, instruction, plan, skill and its labels, time, episode and step, outcome of the episode,
             provenance (demonstration, own rollout or correction), who labelled it
use counters how often the moment was retrieved, and when last
```

![The session store: what happens at every call, and when an episode ends](detailed/session_store_overview.svg)

**Why these changes:**

- A learned image key (DINOv2, SigLIP: networks that turn a picture into a list of numbers) does not depend on object colours,
  which a second task or a real robot would need.
- Filters let one store serve many sessions and tasks without the moments of one misleading another.
- Provenance lets a store that grows from the robot's own episodes be checked and pruned.
- A cap keeps the store from growing without bound. Above it, the oldest moments go first (FIFO), or the least retrieved and,
  among those, the oldest (use and recency).
- A new episode is appended to the database. No file is rewritten.

**The write path.** During an episode, the pictures and the robot's gripper signals of every monitored call are kept. When
the episode ends, the calls are labelled in hindsight by the rules writer, replayed over the robot's own signals (no simulator
needed). Calls after the task is complete are left out, and so are near-duplicates within an episode. The rest is inserted
with the database's keys, and then the cap, if there is one, is applied.

**What it cannot do:**

- **It did not raise the task success.** The live arms completed 88 to 94 of 100 episodes. The same writer with no store at
  all completed 94 in an earlier run on other seeds, so this is only a rough comparison. What the store buys is fewer
  questions to the planning model.
- **The learned image keys were worse than the hand-made colour key** at finding the right skill, also when the camera had
  moved. The colour key in turn needs clearly coloured objects; another scene may need a learned key (argued, not tested
  here).
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
  `google/siglip-base-patch16-224`. They run on the CPU (about 20 to 60 ms per picture) or on a GPU.
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

Measured with the sub-goal policy on the reference task (two blocks, picked up and put back in order) and the VLM writer with
the vote; 100 episodes per live arm, 400 steps each, a store that starts empty. We asked three questions: does a learned key
find moments of the right skill when the exterior camera has moved 5 cm; does the monitor keep its precision and its number of
questions when unrelated sessions write into the same store; and is a capped store as good as an unbounded one? We also
measured what a query and an insert cost as the store grows to 100,000 moments.

- **Which key (offline).** The colour key finds the right skill more often than the learned image keys. The measure is
  *recall@1*: the share of calls for which the single nearest stored moment belongs to the right skill. With the exterior
  camera moved 5 cm it is 0.47 for the colour key, 0.19 for DINOv2 and 0.19 for SigLIP; with the camera where it was, 0.57,
  0.38 and 0.57. The skill of the nearest moment changes in 39 % of the calls for DINOv2 from video compression alone (for the
  colour key in 1.4 %). The live runs therefore all use the colour key.
- **Moments from other sessions (live).** With 20 episodes each of two other instructions (the red block only, the blue block
  first) writing into the store, 28 % of the retrieved moments came from those sessions. The monitor's "done" precision, the
  share of its "done" answers that were right, stayed at 0.99 without a filter and at 0.98 with the filter "same instruction".
  Those moments show the same skills with their correct outcome, so for this task the filter is not needed. It matters where
  the same picture means a different answer under another instruction.
- **Size cap (live).** A cap of 1,000 moments (about 24 episodes) costs no task success: 91 of 100 full tasks with
  use-and-recency eviction, 94 with FIFO, 91 without a cap. Once the cap binds, FIFO makes the monitor ask more
  questions than the uncapped store (more in 47 episodes, fewer in 22: a split that would be very unlikely if both asked
  equally many). Use-and-recency does not (more in 41, fewer in 30: within noise).
- **What the store does.** After about 20 episodes the vote decides a third of the monitor's checks without asking the
  planning model (0.17 to 0.23 of the checks in the first 10 episodes, 0.31 to 0.42 later). The questions per episode fall
  from about 32 to 36 to about 25 to 30, and no episode ended on a false completion.
- **Cost.** The stores for these timings, on a CPU shared with other jobs, were grown synthetically from the 801 recorded
  moments (their keys with a little noise). Putting an episode's 40 moments into the database takes about 15 ms once their
  keys and pictures are ready. A search takes 10 ms at 1,000 moments, 25 ms at 10,000 and 212 ms at 100,000 with exact search
  (with a filter 16, 34 and 302 ms), not counting the query's key. With an index on the DINOv2 key it stays at 15 to 17 ms and
  finds every one of the 6 nearest. Computing one query's key on the CPU takes 3 ms for colour, 58 ms for DINOv2 and 207 ms
  for SigLIP. On disk the store takes about 0.12 MB per moment at 100,000 moments (11.6 GB). In the live runs, the write at
  the end of an episode, with all three keys computed on the CPU, took a median 10.7 s, nearly all of it the keys. It runs
  when the next episode begins.
