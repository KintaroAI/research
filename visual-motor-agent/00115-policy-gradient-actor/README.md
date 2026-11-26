# vm-00115: The Move Token as a Policy-Gradient Actor

**Status:** Running
**Plan:** `tasks/00128-policy-gradient-actor.md` (option (c) of task 00123)
**Source:** `dev/` (runs in progress)

The move token is trained as an actor by REINFORCE on real windows, with our label as the critic: the move made is
one sample of it (no instinct), its advantage is the mean predicted label of 16 sampled moves minus the move's own, and
the move token's cross-entropy on the newest windows is weighted by that advantage, plus an entropy bonus
(`online.actor_pg = 1`, `pg_ent` .03). Arms: PG in the food world against TT (vm-00101), PB in Breakout stage 1
against BD (vm-00103); 12 seeds, 60k ticks, then frozen at 30k and 60k. Readings, set before the results, are in the
task file.

Pilot (3 seeds, 30k ticks, food world): neither `pg_ent` .03 nor .1 learned to forage (food about 10 per 1k at 20–30k,
TT 57); the move token's entropy fell to a near-fixed move.
