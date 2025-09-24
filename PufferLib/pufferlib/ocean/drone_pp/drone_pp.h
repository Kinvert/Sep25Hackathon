// Originally made by Sam Turner and Finlay Sanders, 2025.
// Included in pufferlib under the original project's MIT license.
// https://github.com/stmio/drone

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "raylib.h"
#include "dronelib.h"

#define TASK_IDLE 0
#define TASK_HOVER 1
#define TASK_ORBIT 2
#define TASK_FOLLOW 3
#define TASK_CUBE 4
#define TASK_CONGO 5
#define TASK_FLAG 6
#define TASK_RACE 7
#define TASK_PP2 8
#define TASK_N 9

#define DEBUG 0

char* TASK_NAMES[TASK_N] = {
    "Idle", "Hover", "Orbit", "Follow",
    "Cube", "Congo", "FLAG", "Race", "PP2"
};

#define R (Color){255, 0, 0, 255}
#define W (Color){255, 255, 255, 255}
#define B (Color){0, 0, 255, 255}
Color FLAG_COLORS[64] = {
    B, B, B, B, R, R, R, R,
    B, B, B, B, W, W, W, W,
    B, B, B, B, R, R, R, R,
    B, B, B, B, W, W, W, W,
    R, R, R, R, R, R, R, R,
    W, W, W, W, W, W, W, W,
    R, R, R, R, R, R, R, R,
    W, W, W, W, W, W, W, W
};
#undef R
#undef W
#undef B

typedef struct Client Client;
struct Client {
    Camera3D camera;
    float width;
    float height;

    float camera_distance;
    float camera_azimuth;
    float camera_elevation;
    bool is_dragging;
    Vector2 last_mouse_pos;

    // Trailing path buffer (for rendering only)
    Trail* trails;
};

typedef struct {
    float *observations;
    float *actions;  // Now 2D velocity commands instead of 4 motor commands
    float *rewards;
    unsigned char *terminals;

    float dist;

    Log log;
    int tick;
    int report_interval;
    bool render;

    int task;
    int num_agents;
    Drone* agents;

    int max_rings;
    Ring* ring_buffer;

    int debug;

    float reward_min_dist;
    float reward_max_dist;
    float dist_decay;
    float reward_dist;

    float w_position;
    float w_velocity;
    float w_stability;
    float w_approach;
    float w_hover;

    float pos_const;
    float pos_penalty;

    float grip_k;
    float grip_k_min;
    float grip_k_max;
    float grip_k_decay;

    // PID Controller fields (per agent) - matching Isaac Sim RobustAAC
    float *target_altitudes;     // Target altitude (5m default)
    float *alt_integrals;       // Altitude integral terms
    float *prev_alt_errors;     // Previous altitude errors for derivative
    float *hover_throttles;     // Hover throttle for each agent
    float *motor_commands;      // Actual motor commands computed by PID (for rendering)

    // PID gains matching Isaac Sim
    float kp_alt;    // Altitude proportional gain (0.12)
    float ki_alt;    // Altitude integral gain (0.02)
    float kd_alt;    // Altitude derivative gain (0.18)
    float kp_att;    // Attitude proportional gain (0.35)
    float kd_att;    // Attitude derivative gain (0.08)
    float kp_vel;    // Velocity control gain (0.05)
    float max_tilt;  // Maximum tilt angle (0.05 rad ~3 degrees)

    Client *client;
} DronePP;

void init(DronePP *env) {
    env->render = false;
    env->agents = calloc(env->num_agents, sizeof(Drone));
    env->ring_buffer = calloc(env->max_rings, sizeof(Ring));
    env->log = (Log){0};
    env->tick = 0;

    // Initialize PID controller arrays
    env->target_altitudes = calloc(env->num_agents, sizeof(float));
    env->alt_integrals = calloc(env->num_agents, sizeof(float));
    env->prev_alt_errors = calloc(env->num_agents, sizeof(float));
    env->hover_throttles = calloc(env->num_agents, sizeof(float));
    env->motor_commands = calloc(env->num_agents * 4, sizeof(float));  // 4 motors per agent

    // Set default PID gains (matching Isaac Sim RobustAAC)
    env->kp_alt = 0.12f;
    env->ki_alt = 0.02f;
    env->kd_alt = 0.18f;
    env->kp_att = 0.35f;
    env->kd_att = 0.08f;
    env->kp_vel = 0.05f;
    env->max_tilt = 0.05f;  // ~3 degrees max tilt

    // Set default target altitude for all agents
    for (int i = 0; i < env->num_agents; i++) {
        env->target_altitudes[i] = 5.0f;  // 5m default altitude
        env->alt_integrals[i] = 0.0f;
        env->prev_alt_errors[i] = 0.0f;

        // Calculate hover throttle for each agent
        Drone *agent = &env->agents[i];
        float hover_force = agent->params.mass * agent->params.gravity;
        float hover_force_per_motor = hover_force / 4.0f;
        float hover_rpm = sqrtf(hover_force_per_motor / agent->params.k_thrust);
        float hover_throttle = (hover_rpm / agent->params.max_rpm) * 2.0f - 1.0f;
        env->hover_throttles[i] = clampf(hover_throttle, -1.0f, 1.0f);
    }
}

void add_log(DronePP *env, int idx, bool oob) {
    Drone *agent = &env->agents[idx];
    env->log.score += agent->score;
    env->log.episode_return += agent->episode_return;
    env->log.episode_length += agent->episode_length;
    env->log.collision_rate += agent->collisions / (float)agent->episode_length;
    env->log.perf += agent->score / (float)agent->episode_length;
    if (oob) {
        env->log.oob += 1.0f;
    }
    env->log.n += 1.0f;

    agent->episode_length = 0;
    agent->episode_return = 0.0f;
}

Drone* nearest_drone(DronePP* env, Drone *agent) {
    float min_dist = 999999.0f;
    Drone *nearest = NULL;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *other = &env->agents[i];
        if (other == agent) {
            continue;
        }
        float dx = agent->state.pos.x - other->state.pos.x;
        float dy = agent->state.pos.y - other->state.pos.y;
        float dz = agent->state.pos.z - other->state.pos.z;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = other;
        }
    }
    if (nearest == NULL) {
        int x = 0;

    }
    return nearest;
}

void compute_observations(DronePP *env) {
    int idx = 0;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];

        Quat q_inv = quat_inverse(agent->state.quat);
        Vec3 linear_vel_body = quat_rotate(q_inv, agent->state.vel);
        Vec3 drone_up_world = quat_rotate(agent->state.quat, (Vec3){0.0f, 0.0f, 1.0f});

        // TODO: Need abs observations now right? idk
        // 42
        env->observations[idx++] = linear_vel_body.x / agent->params.max_vel;
        env->observations[idx++] = linear_vel_body.y / agent->params.max_vel;
        env->observations[idx++] = linear_vel_body.z / agent->params.max_vel;
        env->observations[idx++] = clampf(agent->state.vel.x, -1.0f, 1.0f);
        env->observations[idx++] = clampf(agent->state.vel.y, -1.0f, 1.0f);
        env->observations[idx++] = clampf(agent->state.vel.z, -1.0f, 1.0f);

        env->observations[idx++] = agent->state.omega.x / agent->params.max_omega;
        env->observations[idx++] = agent->state.omega.y / agent->params.max_omega;
        env->observations[idx++] = agent->state.omega.z / agent->params.max_omega;

        env->observations[idx++] = drone_up_world.x;
        env->observations[idx++] = drone_up_world.y;
        env->observations[idx++] = drone_up_world.z;

        env->observations[idx++] = agent->state.quat.w;
        env->observations[idx++] = agent->state.quat.x;
        env->observations[idx++] = agent->state.quat.y;
        env->observations[idx++] = agent->state.quat.z;

        env->observations[idx++] = agent->state.rpms[0] / agent->params.max_rpm;
        env->observations[idx++] = agent->state.rpms[1] / agent->params.max_rpm;
        env->observations[idx++] = agent->state.rpms[2] / agent->params.max_rpm;
        env->observations[idx++] = agent->state.rpms[3] / agent->params.max_rpm;

        // ISAAC SIM COMPATIBLE OBSERVATION SPACE
        // Position relative to environment origin (0,0,0) - matches Isaac Sim
        env->observations[idx++] = agent->state.pos.x / GRID_X;
        env->observations[idx++] = agent->state.pos.y / GRID_Y;
        env->observations[idx++] = agent->state.pos.z / GRID_Z;

        // Target relative position - matches Isaac Sim dxyz calculation
        float dx = agent->drop_pos.x - agent->state.pos.x;
        float dy = agent->drop_pos.y - agent->state.pos.y;
        float dz = agent->drop_pos.z - agent->state.pos.z;
        env->observations[idx++] = clampf(dx, -1.0f, 1.0f);
        env->observations[idx++] = clampf(dy, -1.0f, 1.0f);
        env->observations[idx++] = clampf(dz, -1.0f, 1.0f);
        env->observations[idx++] = dx / GRID_X;
        env->observations[idx++] = dy / GRID_Y;
        env->observations[idx++] = dz / GRID_Z;

        // Last rewards - Isaac Sim uses zeros for unused rewards
        env->observations[idx++] = 0.0f; // last_collision_reward (unused in Isaac Sim)
        env->observations[idx++] = 0.0f; // last_target_reward (unused in Isaac Sim)
        env->observations[idx++] = 0.0f; // last_abs_reward (unused in Isaac Sim)

        // Nearest drone relative position - matches Isaac Sim (single agent = zeros)
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;
        env->observations[idx++] = 0.0f;

        // ALWAYS use PP2 task observations to match Isaac Sim
        // Isaac Sim: to_box_b (body frame box position)
        Vec3 to_box = quat_rotate(q_inv, sub3(agent->box_pos, agent->state.pos));
        env->observations[idx++] = to_box.x / GRID_X;
        env->observations[idx++] = to_box.y / GRID_Y;
        env->observations[idx++] = to_box.z / GRID_Z;

        // Isaac Sim: to_drop_b (body frame drop position)
        Vec3 to_drop = quat_rotate(q_inv, sub3(agent->drop_pos, agent->state.pos));
        env->observations[idx++] = to_drop.x / GRID_X;
        env->observations[idx++] = to_drop.y / GRID_Y;
        env->observations[idx++] = to_drop.z / GRID_Z;

        // Isaac Sim: PP2 flag (always 1.0)
        env->observations[idx++] = 1.0f;
    }
}

void move_target(DronePP* env, Drone *agent) {
    agent->target_pos.x += agent->target_vel.x;
    agent->target_pos.y += agent->target_vel.y;
    agent->target_pos.z += agent->target_vel.z;
    if (agent->target_pos.x < -GRID_X || agent->target_pos.x > GRID_X) {
        agent->target_vel.x = -agent->target_vel.x;
    }
    if (agent->target_pos.y < -GRID_Y || agent->target_pos.y > GRID_Y) {
        agent->target_vel.y = -agent->target_vel.y;
    }
    if (agent->target_pos.z < -GRID_Z || agent->target_pos.z > GRID_Z) {
        agent->target_vel.z = -agent->target_vel.z;
    }
    agent->hidden_pos = agent->target_pos;
    agent->hidden_vel = agent->target_vel;
}

void set_target_idle(DronePP* env, int idx) {
    Drone *agent = &env->agents[idx];
    agent->target_pos = (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y), rndf(-MARGIN_Z, MARGIN_Z)};
    agent->target_vel = (Vec3){rndf(-V_TARGET, V_TARGET), rndf(-V_TARGET, V_TARGET), rndf(-V_TARGET, V_TARGET)};
}

void set_target_hover(DronePP* env, int idx) {
    Drone *agent = &env->agents[idx];
    agent->target_pos = agent->state.pos;
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_orbit(DronePP* env, int idx) {
    // Fibbonacci sphere algorithm
    float R = 8.0f;
    float phi = PI * (sqrt(5.0f) - 1.0f);
    float y = 1.0f - 2*((float)idx / (float)env->num_agents);
    float radius = sqrtf(1.0f - y*y);

    float theta = phi * idx;

    float x = cos(theta) * radius;
    float z = sin(theta) * radius;

    Drone *agent = &env->agents[idx];
    agent->target_pos = (Vec3){R*x, R*z, R*y}; // convert to z up 
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_follow(DronePP* env, int idx) {
    Drone* agent = &env->agents[idx];
    if (idx == 0) {
        set_target_idle(env, idx);
    } else {
        agent->target_pos = env->agents[0].target_pos;
        agent->target_vel = env->agents[0].target_vel;
    }
}

void set_target_cube(DronePP* env, int idx) {
    Drone* agent = &env->agents[idx];
    float z = idx / 16;
    idx = idx % 16;
    float x = (float)(idx % 4);
    float y = (float)(idx / 4);
    agent->target_pos = (Vec3){4*x - 6, 4*y - 6, 4*z - 6};
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_congo(DronePP* env, int idx) {
    if (idx == 0) {
        set_target_idle(env, idx);
        return;
    }
    Drone* follow = &env->agents[idx - 1];
    Drone* lead = &env->agents[idx];
    lead->target_pos = follow->target_pos;
    lead->target_vel = follow->target_vel;

    // TODO: Slow hack
    for (int i = 0; i < 40; i++) {
        move_target(env, lead);
    }
}

void set_target_flag(DronePP* env, int idx) {
    Drone* agent = &env->agents[idx];
    float x = (float)(idx % 8);
    float y = (float)(idx / 8);
    x = 2.0f*x - 7;
    y = 5 - 1.5f*y;
    agent->target_pos = (Vec3){0.0f, x, y};
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_race(DronePP* env, int idx) {
    Drone* agent = &env->agents[idx];
    agent->target_pos = env->ring_buffer[agent->ring_idx].pos;
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_pp(DronePP* env, int idx) {
    Drone* agent = &env->agents[idx];
    if (!agent->gripping) {
        agent->target_pos = (Vec3){agent->box_pos.x, agent->box_pos.y, agent->box_pos.z + 1.5f};
    } else {
        agent->target_pos = (Vec3){agent->drop_pos.x, agent->drop_pos.y, agent->drop_pos.z + 1.5f};
    }
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target_pp2(DronePP* env, int idx) {
    Drone* agent = &env->agents[idx];
    if (!agent->gripping) {
        agent->target_pos = (Vec3){agent->box_pos.x, agent->box_pos.y, agent->box_pos.z};
    } else {
        agent->target_pos = (Vec3){agent->drop_pos.x, agent->drop_pos.y, agent->drop_pos.z};
    }
    agent->target_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void set_target(DronePP* env, int idx) {
    if (env->task == TASK_IDLE) {
        set_target_idle(env, idx);
    } else if (env->task == TASK_HOVER) {
        set_target_hover(env, idx);
    } else if (env->task == TASK_ORBIT) {
        set_target_orbit(env, idx);
    } else if (env->task == TASK_FOLLOW) {
        set_target_follow(env, idx);
    } else if (env->task == TASK_CUBE) {
        set_target_cube(env, idx);
    } else if (env->task == TASK_CONGO) {
        set_target_congo(env, idx);
    } else if (env->task == TASK_FLAG) {
        set_target_flag(env, idx);
    } else if (env->task == TASK_RACE) {
        set_target_race(env, idx);
    } else if (env->task == TASK_PP2) {
        set_target_pp2(env, idx);
    }
}

float compute_reward(DronePP* env, Drone *agent, bool collision) {
    if (DEBUG > 0) printf("  Compute Reward\n");
    Vec3 tgt = agent->target_pos;
    if (env->task == TASK_PP2) tgt = agent->hidden_pos;

    Vec3 pos_error = {agent->state.pos.x - tgt.x, agent->state.pos.y - tgt.y, agent->state.pos.z - tgt.z};
    float dist = sqrtf(pos_error.x * pos_error.x + pos_error.y * pos_error.y + pos_error.z * pos_error.z) + 0.00000001;

    Vec3 vel_error = {agent->state.vel.x, agent->state.vel.y, agent->state.vel.z - agent->hidden_vel.z};
    float vel_magnitude = sqrtf(vel_error.x * vel_error.x + vel_error.y * vel_error.y + vel_error.z * vel_error.z);

    float angular_vel_magnitude = sqrtf(agent->state.omega.x * agent->state.omega.x +
                                      agent->state.omega.y * agent->state.omega.y +
                                      agent->state.omega.z * agent->state.omega.z);

    float proximity_factor = clampf(1.0f - dist / env->reward_dist, 0.0f, 1.0f);

    env->reward_dist = clampf(env->tick * -env->dist_decay + env->reward_max_dist, env->reward_min_dist, 100.0f);

    float position_reward = clampf(expf(-dist / (env->reward_dist * env->pos_const)), -env->pos_penalty, 1.0f);

    // slight reward for 0.05 for example, large penalty for over 0.4
    float velocity_penalty = clampf(proximity_factor * (2.0f * expf(-(vel_magnitude - 0.05f) * 10.0f) - 1.0f), -1.0f, 1.0f);
    if (DEBUG > 0) printf("    velocity_penalty = %.3f\n", velocity_penalty);

    float stability_reward = -angular_vel_magnitude / agent->params.max_omega;

    Vec3 to_target_unit = {0, 0, 0};
    if (dist > 0.001f) {
        to_target_unit.x = -pos_error.x / dist;
        to_target_unit.y = -pos_error.y / dist;
        to_target_unit.z = -pos_error.z / dist;
    }
    float approach_dot = to_target_unit.x * agent->state.vel.x +
                        to_target_unit.y * agent->state.vel.y +
                        to_target_unit.z * agent->state.vel.z;

    float approach_weight = clampf(dist / env->reward_dist, 0.0f, 1.0f); // todo
    float approach_reward = approach_weight * clampf(approach_dot / agent->params.max_vel, -0.5f, 0.5f);

    float hover_bonus = 0.0f; // todo add a K
    if (dist < env->reward_dist * 0.2f && vel_magnitude < 0.2f) {
        hover_bonus = 0.5f;
    }

    float collision_penalty = 0.0f;
    if (collision && env->num_agents > 1) {
        Drone *nearest = nearest_drone(env, agent);
        float dx = agent->state.pos.x - nearest->state.pos.x;
        float dy = agent->state.pos.y - nearest->state.pos.y;
        float dz = agent->state.pos.z - nearest->state.pos.z;
        float min_dist = sqrtf(dx*dx + dy*dy + dz*dz);
        if (min_dist < 1.0f) {
            collision_penalty = -1.0f;
            agent->collisions += 1.0f;
        }
    }

    float total_reward = env->w_position * position_reward +
                        env->w_velocity * velocity_penalty +
                        env->w_stability * stability_reward +
                        env->w_approach * approach_reward +
                        env->w_hover * hover_bonus +
                        collision_penalty;

    total_reward = clampf(total_reward, -1.0f, 1.0f);

    float delta_reward = total_reward - agent->last_abs_reward;

    agent->last_collision_reward = collision_penalty;
    agent->last_target_reward = position_reward;
    agent->last_abs_reward = total_reward;
    agent->episode_length++;
    agent->score += total_reward;
    env->dist = dist * dist;
    agent->jitter = 10.0f - (dist + vel_magnitude + angular_vel_magnitude);

    return delta_reward;
}

void reset_pp2(DronePP* env, Drone *agent, int idx) {
    agent->box_pos = (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y), -GRID_Z + 0.5f};
    agent->drop_pos = (Vec3){rndf(-MARGIN_X, MARGIN_X), rndf(-MARGIN_Y, MARGIN_Y), -GRID_Z + 0.5f};
    agent->gripping = false;
    agent->delivered = false;
    agent->grip_height = 0.0f;
    agent->approaching_pickup = false;
    agent->hovering_pickup = false;
    agent->descent_pickup = false;
    agent->approaching_drop = false;
    agent->hovering_drop = false;
    agent->descent_drop = false;
    agent->hover_timer = 0.0f;
    agent->target_pos = agent->box_pos;
    agent->hidden_pos = agent->target_pos;
    agent->hidden_pos.z += 1.0f;
    agent->hidden_vel = (Vec3){0.0f, 0.0f, 0.0f};
}

void reset_agent(DronePP* env, Drone *agent, int idx) {
    agent->episode_return = 0.0f;
    agent->episode_length = 0;
    agent->collisions = 0.0f;
    agent->score = 0.0f;
    agent->ring_idx = 0;
    agent->perfect_grip = false;
    agent->perfect_deliv = false;
    agent->has_delivered = false;
    agent->jitter = 100.0f;

    //float size = 0.2f;
    //init_drone(agent, size, 0.0f);
    float size = rndf(0.1f, 0.4);
    init_drone(agent, size, 0.1f);
    agent->color = FLAG_COLORS[idx];
    agent->color = (Color){255, 0, 0, 255};

    agent->state.pos = (Vec3){
        rndf(-MARGIN_X, MARGIN_X),
        rndf(-MARGIN_Y, MARGIN_Y),
        rndf(-MARGIN_Z, MARGIN_Z)
    };
    agent->prev_pos = agent->state.pos;
    agent->spawn_pos = agent->state.pos;

    if (env->task == TASK_PP2) {
        reset_pp2(env, agent, idx);
    }

    compute_reward(env, agent, env->task != TASK_RACE);
}

void c_reset(DronePP *env) {
    env->tick = 0;
    //env->task = rand() % (TASK_N - 1);

    if (rand() % 4) {
        env->task = TASK_PP2; //CHOOSE TASK
    } else {
        env->task = rand() % (TASK_N - 1);
    }
    env->task = TASK_PP2;

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        reset_agent(env, agent, i);
        set_target(env, i);

        // Reset PID controller states
        env->alt_integrals[i] = 0.0f;
        env->prev_alt_errors[i] = 0.0f;
        env->target_altitudes[i] = 5.0f;  // Reset to 5m default altitude

        // Recalculate hover throttle in case agent parameters changed
        float hover_force = agent->params.mass * agent->params.gravity;
        float hover_force_per_motor = hover_force / 4.0f;
        float hover_rpm = sqrtf(hover_force_per_motor / agent->params.k_thrust);
        float hover_throttle = (hover_rpm / agent->params.max_rpm) * 2.0f - 1.0f;
        env->hover_throttles[i] = clampf(hover_throttle, -1.0f, 1.0f);
    }

    for (int i = 0; i < env->max_rings; i++) {
        Ring *ring = &env->ring_buffer[i];
        *ring = (Ring){0};
    }
    if (env->task == TASK_RACE) {
        float ring_radius = 2.0f;
        reset_rings(env->ring_buffer, env->max_rings, ring_radius);

        // start drone at least MARGIN away from the first ring
        for (int i = 0; i < env->num_agents; i++) {
            Drone *drone = &env->agents[i];
            do {
                drone->state.pos = (Vec3){
                    rndf(-MARGIN_X, MARGIN_X), 
                    rndf(-MARGIN_Y, MARGIN_Y), 
                    rndf(-MARGIN_Z, MARGIN_Z)
                };
            } while (norm3(sub3(drone->state.pos, env->ring_buffer[0].pos)) < 2.0f*ring_radius);
        }
    }
 
    compute_observations(env);
}

// PID Controller that converts 2D velocity commands to 4 motor commands
// Matches Isaac Sim RobustAAC implementation
void pid_control(DronePP *env, int agent_idx, float vx_cmd, float vy_cmd, float* motor_actions) {
    Drone *agent = &env->agents[agent_idx];

    // Get current state
    float current_altitude = agent->state.pos.z;
    float target_altitude = env->target_altitudes[agent_idx];

    // Get velocity in body frame
    Quat q_inv = quat_inverse(agent->state.quat);
    Vec3 vel_body = quat_rotate(q_inv, agent->state.vel);
    float climb_rate = agent->state.vel.z;

    // Get drone's up vector in world frame
    Vec3 up_body = {0.0f, 0.0f, 1.0f};
    Vec3 up_world = quat_rotate(agent->state.quat, up_body);

    // Calculate roll and pitch from quaternion
    float roll = atan2f(2.0f * (agent->state.quat.w * agent->state.quat.x +
                                agent->state.quat.y * agent->state.quat.z),
                       1.0f - 2.0f * (agent->state.quat.x * agent->state.quat.x +
                                     agent->state.quat.y * agent->state.quat.y));
    float pitch = asinf(clampf(2.0f * (agent->state.quat.w * agent->state.quat.y -
                               agent->state.quat.z * agent->state.quat.x), -1.0f, 1.0f));

    // ALTITUDE CONTROL (PID)
    float alt_error = target_altitude - current_altitude;

    // Update integral (with anti-windup)
    env->alt_integrals[agent_idx] += alt_error * DT;
    env->alt_integrals[agent_idx] = clampf(env->alt_integrals[agent_idx], -2.0f, 2.0f);

    // Calculate derivative
    float alt_error_rate = (alt_error - env->prev_alt_errors[agent_idx]) / DT;
    env->prev_alt_errors[agent_idx] = alt_error;

    // PID altitude control
    float altitude_correction = env->kp_alt * alt_error +
                               env->ki_alt * env->alt_integrals[agent_idx] -
                               env->kd_alt * climb_rate;
    altitude_correction = clampf(altitude_correction, -0.15f, 0.12f);

    // Base thrust
    float base_thrust = env->hover_throttles[agent_idx] + altitude_correction;

    // VELOCITY CONTROL (converts velocity commands to desired tilt)
    float desired_roll = -vy_cmd * env->kp_vel;  // Negative for correct direction
    float desired_pitch = vx_cmd * env->kp_vel;

    // Limit desired tilt angles
    desired_roll = clampf(desired_roll, -env->max_tilt, env->max_tilt);
    desired_pitch = clampf(desired_pitch, -env->max_tilt, env->max_tilt);

    // ATTITUDE CONTROL (PD controller)
    float roll_error = desired_roll - roll;
    float pitch_error = desired_pitch - pitch;

    // Angular velocities in body frame
    float roll_rate = agent->state.omega.x;
    float pitch_rate = agent->state.omega.y;

    // PD attitude control
    float roll_correction = env->kp_att * roll_error - env->kd_att * roll_rate;
    float pitch_correction = env->kp_att * pitch_error - env->kd_att * pitch_rate;

    // Clamp corrections
    roll_correction = clampf(roll_correction, -0.3f, 0.3f);
    pitch_correction = clampf(pitch_correction, -0.3f, 0.3f);

    // Tilt protection - reduce thrust if tilted too much
    float tilt_angle = acosf(clampf(up_world.z, -1.0f, 1.0f));
    if (tilt_angle > 0.52f) {  // 30 degrees
        float tilt_factor = fmaxf(0.5f, 1.0f - (tilt_angle - 0.52f) / 0.52f);
        base_thrust *= tilt_factor;
    }

    // Motor mixing: [Front(+Y), Right(+X), Back(-Y), Left(-X)]
    motor_actions[0] = base_thrust + pitch_correction;  // Front motor
    motor_actions[1] = base_thrust + roll_correction;   // Right motor
    motor_actions[2] = base_thrust - pitch_correction;  // Back motor
    motor_actions[3] = base_thrust - roll_correction;   // Left motor

    // Final safety clipping
    for (int i = 0; i < 4; i++) {
        motor_actions[i] = clampf(motor_actions[i], -1.0f, 1.0f);
    }

    // Store motor commands for rendering
    for (int j = 0; j < 4; j++) {
        env->motor_commands[agent_idx * 4 + j] = motor_actions[j];
    }
}

void c_step(DronePP *env) {
    env->tick = (env->tick + 1) % HORIZON;
    //env->log.dist = 0.0f;
    //env->log.dist100 = 0.0f;
    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        env->rewards[i] = 0;
        env->terminals[i] = 0;

        // Get 2D velocity commands from the RL agent (instead of 4 motor commands)
        float vx_cmd = env->actions[2*i];      // Forward/backward velocity command
        float vy_cmd = env->actions[2*i + 1];  // Left/right velocity command

        // Scale velocity commands (assuming [-1, 1] input range)
        vx_cmd *= 2.0f;  // Scale to max 2 m/s
        vy_cmd *= 2.0f;  // Scale to max 2 m/s

        // Use PID controller to convert velocity commands to motor actions
        float motor_actions[4];
        pid_control(env, i, vx_cmd, vy_cmd, motor_actions);

        // Move drone with PID-computed motor actions
        move_drone(agent, motor_actions);

        bool out_of_bounds = agent->state.pos.x < -GRID_X || agent->state.pos.x > GRID_X ||
                             agent->state.pos.y < -GRID_Y || agent->state.pos.y > GRID_Y ||
                             agent->state.pos.z < -GRID_Z || agent->state.pos.z > GRID_Z;

        if (!(env->task == TASK_PP2)) move_target(env, agent);

        float reward = 0.0f;
        if (env->task == TASK_RACE) {
            Ring *ring = &env->ring_buffer[agent->ring_idx];
            reward = compute_reward(env, agent, true);
            float passed_ring = check_ring(agent, ring);
            if (passed_ring > 0) {
                agent->ring_idx = (agent->ring_idx + 1) % env->max_rings;
                env->log.rings_passed += 1.0f;
                set_target(env, i);
                compute_reward(env, agent, true);
            }
            reward += passed_ring;
        // =========================================================================================================================================
        // =========================================================================================================================================
        // =========================================================================================================================================
        } else if (env->task == TASK_PP2) {
            if (DEBUG > 0) printf("\n\n===%d===\n", env->tick);
            agent->hidden_pos.x += agent->hidden_vel.x * DT;
            agent->hidden_pos.y += agent->hidden_vel.y * DT;
            agent->hidden_pos.z += agent->hidden_vel.z * DT;
            if (agent->hidden_pos.z < agent->target_pos.z) {
                agent->hidden_pos.z = agent->target_pos.z;
                agent->hidden_vel.z = 0.0f;
            }
            agent->approaching_pickup = true;
            float speed = norm3(agent->state.vel);
            env->grip_k = clampf(env->tick * -env->grip_k_decay + env->grip_k_max, env->grip_k_min, 100.0f);
            float k = env->grip_k;
            if (DEBUG > 0) printf("  PP2\n");
            if (DEBUG > 0) printf("    K = %.3f\n", k);
            if (DEBUG > 0) printf("    Hidden = %.3f %.3f %.3f\n", agent->hidden_pos.x, agent->hidden_pos.y, agent->hidden_pos.z);
            if (DEBUG > 0) printf("    HiddenV = %.3f %.3f %.3f\n", agent->hidden_vel.x, agent->hidden_vel.y, agent->hidden_vel.z);
            if (DEBUG > 0) printf("    speed = %.3f\n", speed);
            if (!agent->gripping) {
                float dist_to_hidden = sqrtf(powf(agent->state.pos.x - agent->hidden_pos.x, 2) +
                                            powf(agent->state.pos.y - agent->hidden_pos.y, 2) +
                                            powf(agent->state.pos.z - agent->hidden_pos.z, 2));
                float xy_dist_to_box = sqrtf(powf(agent->state.pos.x - agent->box_pos.x, 2) +
                                            powf(agent->state.pos.y - agent->box_pos.y, 2));
                float z_dist_above_box = agent->state.pos.z - agent->box_pos.z;

                // Phase 1 Box Hover
                if (!agent->hovering_pickup) {
                    if (DEBUG > 0) printf("  Phase1\n");
                    if (DEBUG > 0) printf("    dist_to_hidden = %.3f\n", dist_to_hidden);
                    if (DEBUG > 0) printf("    xy_dist_to_box = %.3f\n", xy_dist_to_box);
                    if (DEBUG > 0) printf("    z_dist_above_box = %.3f\n", z_dist_above_box);
                    if (dist_to_hidden < 0.4f && speed < 0.4f) {
                        agent->hovering_pickup = true;
                        agent->color = (Color){255, 255, 255, 255}; // White
                    } else {
                        if (!agent->has_delivered) {
                            agent->color = (Color){255, 100, 100, 255}; // Light Red
                        }
                    }
                }

                // Phase 2 Box Descent
                else {
                    agent->descent_pickup = true;
                    agent->hidden_vel = (Vec3){0.0f, 0.0f, -0.1f};
                    if (DEBUG > 0) printf("  GRIP\n");
                    if (DEBUG > 0) printf("    xy_dist_to_box = %.3f\n", xy_dist_to_box);
                    if (DEBUG > 0) printf("    z_dist_above_box = %.3f\n", z_dist_above_box);
                    if (DEBUG > 0) printf("    speed = %.3f\n", speed);
                    if (DEBUG > 0) printf("    agent->state.vel.z = %.3f\n", agent->state.vel.z);
                    if (
                        xy_dist_to_box < k * 0.1f &&
                        z_dist_above_box < k * 0.1f && z_dist_above_box > 0.0f &&
                        speed < k * 0.1f &&
                        agent->state.vel.z > k * -0.05f && agent->state.vel.z < 0.0f
                    ) {
                        if (k < 1.01) {
                            agent->perfect_grip = true;
                            agent->color = (Color){100, 100, 255, 255}; // Light Blue
                        }
                        agent->gripping = true;
                        reward += 1.0f;
                    } else if (dist_to_hidden > 0.4f || speed > 0.4f) {
                        agent->color = (Color){255, 100, 100, 255}; // Light Red
                    }
                }
            } else {

                // Phase 3 Drop Hover
                agent->box_pos = agent->state.pos;
                agent->box_pos.z -= 0.5f;
                agent->target_pos = agent->drop_pos;
                float xy_dist_to_drop = sqrtf(powf(agent->state.pos.x - agent->drop_pos.x, 2) +
                                            powf(agent->state.pos.y - agent->drop_pos.y, 2));
                float z_dist_above_drop = agent->state.pos.z - agent->drop_pos.z;
                if (!agent->hovering_drop) {
                    agent->target_pos = (Vec3){agent->drop_pos.x, agent->drop_pos.y, agent->drop_pos.z + 0.4f};
                    agent->hidden_pos = (Vec3){agent->drop_pos.x, agent->drop_pos.y, agent->drop_pos.z + 1.0f};
                    agent->hidden_vel = (Vec3){0.0f, 0.0f, 0.0f};
                    if (xy_dist_to_drop < k * 0.4f && z_dist_above_drop > 0.7f && z_dist_above_drop < 1.3f) {
                        agent->hovering_drop = true;
                        reward += 0.25;
                        agent->color = (Color){0, 0, 255, 255}; // Blue
                    }
                }

                // Phase 4 Drop Descent
                else {
                    agent->target_pos = agent->drop_pos;
                    agent->hidden_pos.x = agent->drop_pos.x;
                    agent->hidden_pos.y = agent->drop_pos.y;
                    agent->hidden_vel = (Vec3){0.0f, 0.0f, -0.1f};
                    if (xy_dist_to_drop < k * 0.2f && z_dist_above_drop < k * 0.2f) {
                        agent->hovering_pickup = false;
                        agent->gripping = false;
                        agent->hovering_drop = false;
                        reward += 1.0f;
                        agent->delivered = true;
                        agent->has_delivered = true;
                        if (k < 1.01f && agent->perfect_grip) {
                            agent->perfect_deliv = true;
                            agent->color = (Color){0, 255, 0, 255}; // Green
                        }
                        reset_pp2(env, agent, i);
                    }
                }
            }

            reward += compute_reward(env, agent, true);

            for (int i = 0; i < env->num_agents; i++) {
                Drone *a = &env->agents[i];
                env->log.dist += env->dist;
                env->log.dist100 += 100 - env->dist;
                env->log.jitter += a->jitter;
                if (a->approaching_pickup) env->log.to_pickup += 1.0f;
                if (a->hovering_pickup) env->log.ho_pickup += 1.0f;
                if (a->descent_pickup) env->log.de_pickup += 1.0f;
                if (a->gripping) env->log.gripping += 1.0f;
                if (a->delivered) env->log.delivered += 1.0f;
                if (a->perfect_grip && env->grip_k < 1.01f) env->log.perfect_grip += 1.0f;
                if (a->perfect_deliv && env->grip_k < 1.01f && a->perfect_grip) env->log.perfect_deliv += 1.0f;
                if (a->approaching_drop) env->log.to_drop += 1.0f;
                if (a->hovering_drop) env->log.ho_drop += 1.0f;
            }
        // =========================================================================================================================================
        // =========================================================================================================================================
        // =========================================================================================================================================
        } else {
            // Delta reward
            reward = compute_reward(env, agent, true);
        }

        env->rewards[i] += reward;
        agent->episode_return += reward;

        float min_z = -GRID_Z + 0.2f;
        if (agent->gripping) {
            min_z += 0.1;
        }

        if (out_of_bounds || agent->state.pos.z < min_z) {
            env->rewards[i] -= 1;
            env->terminals[i] = 1;
            add_log(env, i, true);
            reset_agent(env, agent, i);
        } else if (env->tick >= HORIZON - 1) {
            env->terminals[i] = 1;
            add_log(env, i, false);
        }
    }
    if (env->tick >= HORIZON - 1) {
        c_reset(env);
    }

    compute_observations(env);
}

void c_close_client(Client *client) {
    CloseWindow();
    free(client);
}

void c_close(DronePP *env) {
    if (env->client != NULL) {
        c_close_client(env->client);
    }
}

static void update_camera_position(Client *c) {
    float r = c->camera_distance;
    float az = c->camera_azimuth;
    float el = c->camera_elevation;

    float x = r * cosf(el) * cosf(az);
    float y = r * cosf(el) * sinf(az);
    float z = r * sinf(el);

    c->camera.position = (Vector3){x, y, z};
    c->camera.target = (Vector3){0, 0, 0};
}

void handle_camera_controls(Client *client) {
    Vector2 mouse_pos = GetMousePosition();

    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = true;
        client->last_mouse_pos = mouse_pos;
    }

    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) {
        client->is_dragging = false;
    }

    if (client->is_dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mouse_delta = {mouse_pos.x - client->last_mouse_pos.x,
                               mouse_pos.y - client->last_mouse_pos.y};

        float sensitivity = 0.005f;

        client->camera_azimuth -= mouse_delta.x * sensitivity;

        client->camera_elevation += mouse_delta.y * sensitivity;
        client->camera_elevation =
            clampf(client->camera_elevation, -PI / 2.0f + 0.1f, PI / 2.0f - 0.1f);

        client->last_mouse_pos = mouse_pos;

        update_camera_position(client);
    }

    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        client->camera_distance -= wheel * 2.0f;
        client->camera_distance = clampf(client->camera_distance, 5.0f, 50.0f);
        update_camera_position(client);
    }
}

Client *make_client(DronePP *env) {
    Client *client = (Client *)calloc(1, sizeof(Client));

    client->width = WIDTH;
    client->height = HEIGHT;

    SetConfigFlags(FLAG_MSAA_4X_HINT); // antialiasing
    InitWindow(WIDTH, HEIGHT, "PufferLib DronePP");

#ifndef __EMSCRIPTEN__
    SetTargetFPS(60);
#endif

    if (!IsWindowReady()) {
        TraceLog(LOG_ERROR, "Window failed to initialize\n");
        free(client);
        return NULL;
    }

    client->camera_distance = 40.0f;
    client->camera_azimuth = 0.0f;
    client->camera_elevation = PI / 10.0f;
    client->is_dragging = false;
    client->last_mouse_pos = (Vector2){0.0f, 0.0f};

    client->camera.up = (Vector3){0.0f, 0.0f, 1.0f};
    client->camera.fovy = 45.0f;
    client->camera.projection = CAMERA_PERSPECTIVE;

    update_camera_position(client);

    // Initialize trail buffer
    client->trails = (Trail*)calloc(env->num_agents, sizeof(Trail));
    for (int i = 0; i < env->num_agents; i++) {
        Trail* trail = &client->trails[i];
        trail->index = 0;
        trail->count = 0;
        for (int j = 0; j < TRAIL_LENGTH; j++) {
            trail->pos[j] = env->agents[i].state.pos;
        }
    }

    return client;
}

const Color PUFF_RED = (Color){187, 0, 0, 255};
const Color PUFF_CYAN = (Color){0, 187, 187, 255};
const Color PUFF_WHITE = (Color){241, 241, 241, 241};
const Color PUFF_BACKGROUND = (Color){6, 24, 24, 255};

void DrawRing3D(Ring ring, float thickness, Color entryColor, Color exitColor) {
    float half_thick = thickness / 2.0f;

    Vector3 center_pos = {ring.pos.x, ring.pos.y, ring.pos.z};

    Vector3 entry_start_pos = {center_pos.x - half_thick * ring.normal.x,
                               center_pos.y - half_thick * ring.normal.y,
                               center_pos.z - half_thick * ring.normal.z};

    DrawCylinderWiresEx(entry_start_pos, center_pos, ring.radius, ring.radius, 32, entryColor);

    Vector3 exit_end_pos = {center_pos.x + half_thick * ring.normal.x,
                            center_pos.y + half_thick * ring.normal.y,
                            center_pos.z + half_thick * ring.normal.z};

    DrawCylinderWiresEx(center_pos, exit_end_pos, ring.radius, ring.radius, 32, exitColor);
}


void c_render(DronePP *env) {
    if (env->client == NULL) {
        env->client = make_client(env);
        if (env->client == NULL) {
            TraceLog(LOG_ERROR, "Failed to initialize client for rendering\n");
            return;
        }
    }
    env->render = true;
    env->grip_k_max = 1.0f;
    env->grip_k_min = 1.0f;
    if (WindowShouldClose()) {
        c_close(env);
        exit(0);
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        c_close(env);
        exit(0);
    }

    if (IsKeyPressed(KEY_SPACE)) {
        env->task = (env->task + 1) % TASK_N;
        for (int i = 0; i < env->num_agents; i++) {
            set_target(env, i);
        }
        if (env->task == TASK_RACE) {
            float ring_radius = 2.0f;
            reset_rings(env->ring_buffer, env->max_rings, ring_radius);
        }
    }

    handle_camera_controls(env->client);

    Client *client = env->client;

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];
        Trail *trail = &client->trails[i];
        trail->pos[trail->index] = agent->state.pos;
        trail->index = (trail->index + 1) % TRAIL_LENGTH;
        if (trail->count < TRAIL_LENGTH) {
            trail->count++;
        }
        if (env->terminals[i]) {
            trail->index = 0;
            trail->count = 0;
        }
    }

    BeginDrawing();
    ClearBackground(PUFF_BACKGROUND);

    BeginMode3D(client->camera);

    // draws bounding cube
    DrawCubeWires((Vector3){0.0f, 0.0f, 0.0f}, GRID_X * 2.0f,
        GRID_Y * 2.0f, GRID_Z * 2.0f, WHITE);

    for (int i = 0; i < env->num_agents; i++) {
        Drone *agent = &env->agents[i];

        // draws drone body
        DrawSphere((Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z}, 0.3f, agent->color);

        // draws rotors according to thrust (using stored motor commands from PID)
        float T[4];
        for (int j = 0; j < 4; j++) {
            float rpm = (env->motor_commands[4*i + j] + 1.0f) * 0.5f * agent->params.max_rpm;
            T[j] = agent->params.k_thrust * rpm * rpm;
        }

        const float rotor_radius = 0.15f;
        const float visual_arm_len = agent->params.arm_len * 4.0f;

        Vec3 rotor_offsets_body[4] = {{+visual_arm_len, 0.0f, 0.0f},
                                      {-visual_arm_len, 0.0f, 0.0f},
                                      {0.0f, +visual_arm_len, 0.0f},
                                      {0.0f, -visual_arm_len, 0.0f}};

        //Color base_colors[4] = {ORANGE, PURPLE, LIME, SKYBLUE};
        Color base_colors[4] = {agent->color, agent->color, agent->color, agent->color};

        for (int j = 0; j < 4; j++) {
            Vec3 world_off = quat_rotate(agent->state.quat, rotor_offsets_body[j]);

            Vector3 rotor_pos = {agent->state.pos.x + world_off.x, agent->state.pos.y + world_off.y,
                                 agent->state.pos.z + world_off.z};

            float rpm = (env->motor_commands[4*i + j] + 1.0f) * 0.5f * agent->params.max_rpm;
            float intensity = 0.75f + 0.25f * (rpm / agent->params.max_rpm);

            Color rotor_color = (Color){(unsigned char)(base_colors[j].r * intensity),
                                        (unsigned char)(base_colors[j].g * intensity),
                                        (unsigned char)(base_colors[j].b * intensity), 255};

            DrawSphere(rotor_pos, rotor_radius, rotor_color);

            DrawCylinderEx((Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z}, rotor_pos, 0.02f, 0.02f, 8,
                           BLACK);
        }

        // draws line with direction and magnitude of velocity / 10
        if (norm3(agent->state.vel) > 0.1f) {
            DrawLine3D((Vector3){agent->state.pos.x, agent->state.pos.y, agent->state.pos.z},
                       (Vector3){agent->state.pos.x + agent->state.vel.x * 0.1f, agent->state.pos.y + agent->state.vel.y * 0.1f,
                                 agent->state.pos.z + agent->state.vel.z * 0.1f},
                       MAGENTA);
        }

        // Draw trailing path
        Trail *trail = &client->trails[i];
        if (trail->count <= 2) {
            continue;
        }
        for (int j = 0; j < trail->count - 1; j++) {
            int idx0 = (trail->index - j - 1 + TRAIL_LENGTH) % TRAIL_LENGTH;
            int idx1 = (trail->index - j - 2 + TRAIL_LENGTH) % TRAIL_LENGTH;
            float alpha = (float)(TRAIL_LENGTH - j) / (float)trail->count * 0.8f; // fade out
            Color trail_color = ColorAlpha((Color){0, 187, 187, 255}, alpha);
            DrawLine3D((Vector3){trail->pos[idx0].x, trail->pos[idx0].y, trail->pos[idx0].z},
                       (Vector3){trail->pos[idx1].x, trail->pos[idx1].y, trail->pos[idx1].z},
                       trail_color);
        }

    }

    // Rings
    if (env->task == TASK_RACE) {
        float ring_thickness = 0.2f;
        for (int i = 0; i < env->max_rings; i++) {
            Ring ring = env->ring_buffer[i];
            DrawRing3D(ring, ring_thickness, GREEN, BLUE);
        }
    }

    if (env->task == TASK_PP2) {
        for (int i = 0; i < env->num_agents; i++) {
            Drone *agent = &env->agents[i];
            Vec3 render_pos = agent->box_pos;
            DrawCube((Vector3){render_pos.x, render_pos.y, render_pos.z}, 0.4f, 0.4f, 0.4f, BROWN);
            DrawCube((Vector3){agent->drop_pos.x, agent->drop_pos.y, agent->drop_pos.z}, 0.5f, 0.5f, 0.1f, YELLOW);
            //DrawSphere((Vector3){agent->hidden_pos.x, agent->hidden_pos.y, agent->hidden_pos.z}, 0.05f, YELLOW);
        }
    }

    if (IsKeyDown(KEY_TAB)) {
        for (int i = 0; i < env->num_agents; i++) {
            Drone *agent = &env->agents[i];
            Vec3 target_pos = agent->target_pos;
            DrawSphere((Vector3){target_pos.x, target_pos.y, target_pos.z}, 0.45f, (Color){0, 255, 255, 100});
        }
    }

    EndMode3D();

    DrawText("Left click + drag: Rotate camera", 10, 10, 16, PUFF_WHITE);
    DrawText("Mouse wheel: Zoom in/out", 10, 30, 16, PUFF_WHITE);
    DrawText(TextFormat("Task: %s", TASK_NAMES[env->task]), 10, 50, 16, PUFF_WHITE);
    DrawText(TextFormat("K = %.3f", env->grip_k), 10, 70, 16, PUFF_WHITE);

    EndDrawing();
}
