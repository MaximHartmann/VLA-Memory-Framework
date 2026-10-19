# The session store: the long-term memory on a vector database

## The idea

The VLM writer decides at every policy call whether the current skill is done. Its long-term memory is a store of
labelled moments of past episodes: the two camera pictures of one moment and what was true then (which skill was active,
where each object was, whether the gripper was closed). At a call, the stored moments that look most like the current
pictures are retrieved; they vote on the question (the memory-vote gate) and the nearest of them are shown to the planning
model as examples. Until now this store is one array file, and "look most like" is measured with a hand-made key: a
coarse colour thumbnail of each picture plus the area and position of each object colour (the colour-blob key).

The session store keeps the same moments in an embedded vector database (LanceDB: a directory of files, no server). One
row is one stored moment:

```
keys         the colour-blob key (as before), a DINOv2 ViT-S/14 embedding of the exterior picture,
             a SigLIP embedding of the exterior and wrist pictures
pictures     the exterior and the wrist picture (PNG)
payload      session, task, instruction, plan, skill and its labels, time, episode and step, outcome of the episode,
             provenance (demonstration, own rollout or correction), who labelled it
use counters how often the moment was retrieved, and when last
```

What the database adds over the array file:

- **Keys.** Several keys live side by side; the run chooses which one retrieves. The learned image embeddings do not
  depend on object colours, which a second task or the real robot needs, and should be less sensitive to a camera that
  is not exactly where it was when the moments were recorded.
- **Filters.** A retrieval can be restricted by the payload (same task, same instruction, same plan, same session), so
  one store can serve many sessions and tasks without the moments of one misleading another.
- **Provenance.** Every moment says where it came from and who labelled it, so a store that grows from the robot's own
  episodes can be audited and pruned.
- **Eviction.** Above a size cap, moments are removed by a rule: oldest first (FIFO), or least retrieved first and among
  those the oldest (use and recency). Without a cap the store grows without bound.
- **Scale.** Inserting a new episode appends to the database instead of rewriting a file, and retrieval time and insert
  time can be measured as the store grows.

The read path is unchanged for the planner: the k nearest moments by the chosen key (with the optional filter) vote and
are shown, at most two from one past episode, with the same vote rule and the same question. The write path is the online
store's: during an episode every monitored call's pictures and the robot's gripper signals are kept; when the episode
ends, the calls are labelled in hindsight by the rules writer replayed over the robot's own signals (no simulator), calls
after the task is complete are left out, near-duplicates within an episode are left out, and the rest is inserted with
all three keys. Then the cap, if any, is applied.

Three questions are asked of it: (1) camera shift: with the exterior camera moved by 5 cm, does the learned key still
retrieve moments of the right skill? (2) interference: when unrelated sessions (other instructions) write into the same
store, does the filtered store keep the monitor's precision and its number of questions? (3) a size cap: is a capped
store with use-and-recency eviction as good as an unbounded one, and FIFO not? And what does a query and an insert cost as
the store grows to 100,000 moments?

## Prerequisites

- The VLM writer with the planning model (see the VLM writer's page), and the environment's gripper signals
  (memory/proprio_steps or the current finger values) for the write path, as for the rules writer.
- Python packages: lancedb and pyarrow (the database), torch and transformers (the two image encoders).
- The encoder weights, downloaded once into the Hugging Face cache: facebook/dinov2-small and
  google/siglip-base-patch16-224. They run on the CPU (about 20 to 60 ms per picture) or on a GPU.
- The session_store package. It is kept beside the framework, not inside it: put its folder on PYTHONPATH. It uses only
  the framework's extension interface and changes nothing in the framework.
- Optional: an exemplar store file (scripts/build_exemplar_store.py) to import as demonstrations; otherwise the store
  starts empty and grows from the robot's own episodes.

## How to use it in the framework

The store is an extension of the proxy. It provides the planner's memory (in place of `--memory`), records the calls and
writes the finished episodes:

```
python -m vla_memory.proxy --writer vlm --method subgoal --listen-port 19661 --upstream-port 18661 \
    --vlm-url http://127.0.0.1:8100/v1 --log run/planner.jsonl \
    --with session_store.extension:SessionStoreExtension \
    --store-dir stores/session_a --store-key dinov2 --store-filter instruction --store-session reference
```

Options:

| option | meaning | default |
|---|---|---|
| `--store-dir DIR` | the database directory; created when missing, resumed when it exists | required |
| `--store-import FILE` | an exemplar store (.npz) imported as demonstrations when the database is new | none |
| `--store-key colour\|dinov2\|siglip` | the key that retrieves | dinov2 |
| `--store-filter none\|task\|instruction\|plan\|session` | retrieve only moments with the same task, instruction text, plan or session | none |
| `--store-session NAME` | the session written with every new moment (and matched by the session filter) | the start time |
| `--store-write` / `--no-store-write` | add the robot's own episodes after each episode | on |
| `--store-labeller rules\|oracle` | who labels the robot's calls: the rules writer on the robot's signals, or the simulator's prompt | rules |
| `--store-cap N` | most moments kept; 0 = unbounded | 0 |
| `--store-evict fifo\|use_recency` | which moments go above the cap | use_recency |
| `--store-device cpu\|cuda` | where the encoders run | cpu |

The gate and the examples keep their own options (`--k`, `--k-vote`, `--gate-min-done`). The planner log's monitor records
note the retrieval time and the store size; one `store_update` record per episode notes what was added, what was evicted,
the time spent and the size on disk. A store directory can be read outside the proxy with
`session_store.store.SessionStore` (open, query by key and filter, insert, evict), for example to re-encode an existing
store with another key and measure recall offline.

## Our findings

Measured with the sub-goal policy on the reference task (two blocks, picked up and placed in order) and the
VLM writer with the gate; the results will be entered here when the experiment has run:

- camera shift (offline): recall@1 and recall@5 of the right skill for the colour-blob, DINOv2 and SigLIP keys, with the
  exterior camera moved by 5 cm and without, on recorded calls against the simulator's tracker;
- interference (live): "done" precision of the monitor, questions per episode and full tasks per block of 10 episodes,
  when 20 episodes each of two unrelated instructions write into the store, for the colour key without a filter and the
  DINOv2 key with the filter "same instruction"; and the share of retrieved moments that came from the unrelated sessions;
- size cap (live): full tasks and questions per episode with a cap of 1,000 moments under use-and-recency eviction, under
  FIFO, and without a cap;
- cost: retrieval time and insert time against the store size from 1,000 to 100,000 moments, and the size on disk.
