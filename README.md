# VLA-Memory-Framework

A framework to incorporate different kinds of knowledge bases and memory into a robotic AI agent built on a
vision-language-action (VLA) policy.

A VLA policy maps the current camera images and a language instruction to robot actions. On its own it is
memoryless: it does not know what it has already done in the current task, it does not learn from earlier tasks, and
it cannot adapt to a user over time. This framework adds that missing part. It defines the places where memory can
live in a VLA agent: the working memory of the running task, long-term memory kept outside the models in a store,
knowledge written into the models' weights, and the symbolic task state. It provides the interfaces to plug a memory
method into the control loop and collects reference implementations, so that different memory methods and knowledge
bases can be built, exchanged and compared against the same policy, the same task and the same metrics.

The framework is model-agnostic: the VLA policy, the language or vision-language model, the task and the memory
method are replaceable components behind small interfaces.

## First approach: the planner loop

The first method realised in the framework is a planner loop. A proxy sits between the environment and the policy
server and relays every observation. A vision-language model (VLM) acts as the planner: it decomposes the user's
instruction once into the skills the policy was trained on, and at every policy call it judges from the camera
images, the robot's proprioception and retrieved examples of labelled past observations whether the current skill is
complete. The skills and their completion rules come from a task specification file. The proxy keeps the plan and the
current step as working memory, a retrieval store of labelled past observations serves as long-term memory, and the
policy only ever receives the prompt of the current skill.

## Implemented methods

| method | memory used | status | documentation |
|---|---|---|---|
| Planner loop with exemplar memory | working memory (plan and current step, held by the proxy); long-term memory (retrieval store of labelled past observations) | implemented and evaluated in simulation | [docs/methods/planner_loop.md](docs/methods/planner_loop/README.md) |

Each further method gets its own page under `docs/methods/` and a row in this table.
