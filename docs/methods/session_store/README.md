# The session store: the long-term memory on a vector database

## The idea

The VLM writer decides at every policy call whether the current skill is done. Its long-term memory is a store of
labelled moments of past episodes: the two camera pictures of one moment and what was true then (which skill was active,
where each object was, whether the gripper was closed). At a call, the stored moments that look most like the current
pictures are retrieved; they vote on the question (the memory-vote gate) and the nearest of them are shown to the planning
model as examples. By default this store is one array file (`--memory`), and "look most like" is measured with a hand-made
key: a coarse colour thumbnail of each picture plus the area and position of each object colour (the colour-blob key).

The session store keeps the same moments in an embedded vector database (LanceDB: a directory of files, no server). One
row is one stored moment:

```
keys         the colour-blob key (always), and optionally a DINOv2 ViT-S/14 embedding of the exterior picture
             and a SigLIP embedding of the exterior and wrist pictures
pictures     the exterior and the wrist picture (PNG)
payload      session, task, instruction, plan, skill and its labels, time, episode and step, outcome of the episode,
             provenance (demonstration, own rollout or correction), who labelled it
use counters how often the moment was retrieved, and when last
```

What the database adds over the array file:

- **Keys.** Several keys can live side by side; the run chooses which one retrieves (the colour-blob key by default). The
  learned image embeddings do not depend on object colours, which a second task or the real robot needs, and should be less
  sensitive to a camera that is not exactly where it was when the moments were recorded.
- **Filters.** A retrieval can be restricted by the payload (same task, same instruction, same plan, same session), so
  one store can serve many sessions and tasks without the moments of one misleading another.
- **Provenance.** Every moment says where it came from and who labelled it, so a store that grows from the robot's own
  episodes can be audited and pruned.
- **Eviction.** Above a size cap, moments are removed by a rule: oldest first (FIFO), or least retrieved first and among
  those the oldest (use and recency). Without a cap the store grows without bound.
- **Scale.** Inserting a new episode appends to the database instead of rewriting a file, and retrieval time and insert
  time can be measured as the store grows.

The read path is the same for the planner as with the array file: the k nearest moments by the chosen key (with the
optional filter) vote and are shown, at most two from one past episode, with the same vote rule and the same question.
The write path: during an episode every monitored call's pictures and the robot's gripper signals are kept; when the
episode ends, the calls are labelled in hindsight by the rules writer replayed over the robot's own signals (no
simulator), calls after the task is complete are left out, near-duplicates within an episode are left out (by the colour
key, whichever key retrieves), and the rest is inserted with the database's keys. Then the cap, if any, is applied.

Three questions are asked of it: (1) camera shift: with the exterior camera moved by 5 cm, does the learned key still
retrieve moments of the right skill? (2) interference: when unrelated sessions (other instructions) write into the same
store, does the filtered store keep the monitor's precision and its number of questions? (3) a size cap: is a capped
store with use-and-recency eviction as good as an unbounded one, and FIFO not? And what does a query and an insert cost as
the store grows to 100,000 moments?

## Prerequisites

- The VLM writer with the planning model (see the VLM writer's page), and the environment's gripper signals
  (memory/proprio_steps or the current finger values) for the write path, as for the rules writer.
- Python packages: lancedb and pyarrow for the database (`pip install -e '.[session_store]'`). The keys dinov2 and siglip
  also need torch and transformers (`pip install -e '.[session_store_encoders]'`); the default colour key needs neither.
- For dinov2 and siglip, the encoder weights, downloaded once into the Hugging Face cache: facebook/dinov2-small and
  google/siglip-base-patch16-224. They run on the CPU (about 20 to 60 ms per picture) or on a GPU.
- Optional: an exemplar store file (scripts/build_exemplar_store.py) to import as demonstrations; otherwise the store
  starts empty and grows from the robot's own episodes.

## How to use it in the framework

The store is an extension of the proxy. It provides the planner's memory (in place of `--memory`), records the calls and
writes the finished episodes:

```
python -m vla_memory.proxy --writer vlm --method subgoal --listen-port 19661 --upstream-port 18661 \
    --vlm-url http://127.0.0.1:8100/v1 --log run/planner.jsonl \
    --with session_store \
    --store-dir stores/session_a --store-filter instruction --store-session reference
```

The colour key retrieves unless `--store-key dinov2` or `siglip` says otherwise. A new database stores the colour key, which
also decides the near-duplicates of an episode, and the retrieval key; `--store-keys` adds more (`--store-keys dinov2 siglip`
for all three). An existing database keeps the keys it was created with and retrieves with any of them; asking for another
is a usage error. Every new moment gets all of the database's keys, so a database with a learned key needs torch and
transformers to grow, even when the colour key retrieves.

Options:

| option | meaning | default |
|---|---|---|
| `--store-dir DIR` | the database directory; created when missing, resumed when it exists | required |
| `--store-import FILE` | an exemplar store (.npz) imported as demonstrations when the database is new | none |
| `--store-import-instruction TEXT` | the instruction recorded with the imported demonstrations | the task file's first instruction example |
| `--store-key colour\|dinov2\|siglip` | the key that retrieves | colour |
| `--store-keys KEY [KEY ...]` | further keys a new database stores with every moment, besides the colour key and the retrieval key | none |
| `--store-filter none\|task\|instruction\|plan\|session` | retrieve only moments with the same task, instruction text, plan or session | none |
| `--store-session NAME` | the session written with every new moment (and matched by the session filter) | the start time |
| `--store-write` / `--no-store-write` | add the robot's own episodes after each episode | on |
| `--store-labeller rules\|oracle` | who labels the robot's calls: the rules writer on the robot's signals, or the simulator's prompt | rules |
| `--store-cap N` | most moments kept; 0 = unbounded | 0 |
| `--store-evict fifo\|use_recency` | which moments go above the cap | use_recency |
| `--store-device cpu\|cuda` | where the encoders of the learned keys run | cpu |
| `--store-threads N` | CPU threads of the encoders | 8 |

The gate and the examples keep their own options (`--k`, `--k-vote`, `--gate-min-done`). The planner log's monitor records
note the retrieval time and the store size; one `store_update` record per episode notes what was added, what was evicted,
the time spent and the size on disk. A store directory can be read outside the proxy with
`vla_memory.methods.session_store.store.SessionStore` (open, query by key and filter, insert, evict), for example to
re-encode an existing store with another key and measure recall offline.

## Our findings

Measured with the sub-goal policy on the reference task (two blocks, picked up and placed in order) and the VLM writer with
the gate; 100 episodes per live arm.

- **Which key (offline).** The colour-blob key finds the right skill more often than the learned image keys: recall@1 0.47
  against 0.19 (DINOv2) and 0.19 (SigLIP) with the exterior camera moved by 5 cm, and 0.57 against 0.38 and 0.57 without the
  shift. Video compression alone changes DINOv2's nearest neighbour in 39 % of the calls. The live runs therefore all use the
  colour key.
- **Moments from other sessions (live).** With 20 episodes each of two other instructions (the red block only, the blue block
  first) writing into the store, 28 % of the retrieved moments came from those sessions, and the monitor's "done" precision
  stayed at 0.99 without a filter (0.98 with the filter "same instruction"). Those moments show the same skills with their
  correct outcome, so the filter is not needed for this task; it matters where the same picture means a different answer
  under another instruction.
- **Size cap (live).** A cap of 1,000 moments (about 24 episodes) costs no task success: 91 of 100 full tasks with
  use-and-recency eviction, 94 with FIFO, 91 without a cap. Once the cap binds, FIFO makes the monitor ask more questions
  than the uncapped store (more in 47 episodes, fewer in 22); use-and-recency does not.
- **What the store does.** After about 20 episodes the memory vote decides a third of the monitor's checks without asking
  the planning model (0.17-0.23 of the checks in the first 10 episodes, 0.31-0.42 later), the questions per episode fall from
  32-36 to 25-30, and no episode ended on a false completion.
- **Cost.** Inserting an episode (40 moments) takes about 15 ms. A query takes 10 ms at 1,000 moments, 25 ms at 10,000 and
  212 ms at 100,000 with flat search (with a filter 16, 34 and 302 ms); with an IVF-PQ index on the DINOv2 key it stays at
  15-17 ms. On disk the store takes about 0.12 MB per moment (11.6 GB at 100,000). Encoding one query on the CPU: colour
  3 ms, DINOv2 58 ms, SigLIP 207 ms.
