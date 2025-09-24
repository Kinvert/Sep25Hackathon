#include "drone_pp.h"

#define Env DronePP
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
    // Isaac Sim compatible - use hardcoded values matching working PID implementation
    env->num_agents = unpack(kwargs, "num_agents");

    // Hardcode all parameters to match Isaac Sim exactly
    env->max_rings = 5;

    // These don't affect PID control but set reasonable defaults
    env->reward_min_dist = 1.6f;
    env->reward_max_dist = 77.0f;
    env->dist_decay = 0.15f;

    env->w_position = 1.13f;
    env->w_velocity = 0.15f;
    env->w_stability = 2.0f;
    env->w_approach = 2.2f;
    env->w_hover = 1.5f;

    env->pos_const = 0.63f;
    env->pos_penalty = 0.03f;

    env->grip_k_min = 1.0f;
    env->grip_k_max = 15.0f;
    env->grip_k_decay = 0.095f;

    init(env);
    return 0;
}

static int my_log(PyObject *dict, Log *log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "rings_passed", log->rings_passed);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "oob", log->oob);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);

    assign_to_dict(dict, "jitter", log->jitter);
    assign_to_dict(dict, "perfect_grip", log->perfect_grip);
    assign_to_dict(dict, "perfect_deliv", log->perfect_deliv);
    assign_to_dict(dict, "to_pickup", log->to_pickup);
    assign_to_dict(dict, "ho_pickup", log->ho_pickup);
    assign_to_dict(dict, "de_pickup", log->de_pickup);
    assign_to_dict(dict, "to_drop", log->to_drop);
    assign_to_dict(dict, "ho_drop", log->ho_drop);
    assign_to_dict(dict, "dist", log->dist);
    assign_to_dict(dict, "dist100", log->dist100);

    assign_to_dict(dict, "n", log->n);
    return 0;
}
