#include "drone_pickplace.h"

#define Env DronePickPlace
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->num_drones = unpack(kwargs, "num_drones");
    env->num_objects = unpack(kwargs, "num_objects");
    env->num_targets = unpack(kwargs, "num_targets");
    env->world_size = unpack(kwargs, "world_size");
    env->max_height = unpack(kwargs, "max_height");
    env->max_steps = unpack(kwargs, "max_steps");

    env->reward_approach = unpack(kwargs, "reward_approach");
    env->reward_complete = unpack(kwargs, "reward_complete");
    env->reward_grasp = unpack(kwargs, "reward_grasp");
    env->reward_place = unpack(kwargs, "reward_place");
    env->penalty_crash = unpack(kwargs, "penalty_crash");
    env->penalty_no_progress = unpack(kwargs, "penalty_no_progress");
    env->penalty_time = unpack(kwargs, "penalty_time");

    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "grasp_success", log->grasp_success);
    assign_to_dict(dict, "placement_success", log->placement_success);
    assign_to_dict(dict, "efficiency", log->efficiency);
    return 0;
}
