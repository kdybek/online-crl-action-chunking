#!/bin/bash

sbatch run.sh --env arm_reach --group manip
sbatch run.sh --env arm_grasp --group manip
sbatch run.sh --env arm_push_easy --group manip
sbatch run.sh --env arm_binpick_easy --group manip
sbatch actd3.sh --env ant_hardest_maze --group actd3
sbatch actd3.sh --env humanoid --group actd3
