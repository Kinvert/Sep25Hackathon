/* Drone Pick and Place: An RL environment for drone manipulation tasks
 * Training drones to pick up objects and place them at target locations
 * For the REVEL x LycheeAI Isaac Sim Hackathon
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "raylib.h"

const uint8_t MOVE_FORWARD = 0;
const uint8_t MOVE_BACKWARD = 1;
const uint8_t MOVE_LEFT = 2;
const uint8_t MOVE_RIGHT = 3;
const uint8_t MOVE_UP = 4;
const uint8_t MOVE_DOWN = 5;
const uint8_t ROTATE_LEFT = 6;
const uint8_t ROTATE_RIGHT = 7;
const uint8_t GRIPPER_OPEN = 8;
const uint8_t GRIPPER_CLOSE = 9;

const uint8_t STATE_SEARCHING = 0;
const uint8_t STATE_APPROACHING = 1;
const uint8_t STATE_GRASPING = 2;
const uint8_t STATE_TRANSPORTING = 3;
const uint8_t STATE_PLACING = 4;
const uint8_t STATE_CRASHED = 5;

typedef struct {
    float perf; // 0-1 normalized performance metric
    float score; // Unnormalized score
    float episode_return;
    float episode_length;
    float grasp_success;
    float placement_success;
    float efficiency; // Path efficiency metric
    float n; // Required as the last field 
} Log;

typedef struct {
    int grasp_attempts;
    int grasp_successes;
    int placement_attempts;
    int placement_successes;
} Stats;

typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float qw, qx, qy, qz; // Orientation quaternion
    float wx, wy, wz;
    float yaw, pitch, roll; // Euler angles for convenience
    float gripper_open; // 0 = closed, 1 = open
    uint8_t state;
    int ticks_without_progress;
} Drone;

typedef struct {
    float x, y, z;
    float vx, vy, vz;
    float radius;
    uint8_t is_grasped;
    uint8_t is_placed;
} Object;

typedef struct {
    float x, y, z;
    float radius;
    uint8_t has_object;
} TargetZone;

typedef struct {
    Camera3D camera;
    int initialized;
} Client;

typedef struct {
    Log log; // Required field
    Stats stats; // Track attempts and successes
    Client* client;
    Drone* drones; // Support multiple drones
    Object* objects;
    TargetZone* targets;
    float* observations; // Required - continuous observations
    int* actions; // Required - discrete actions
    float* rewards; // Required
    uint8_t* terminals; // Required

    int num_drones;
    int num_objects;
    int num_targets;
    float world_size;
    float max_height;
    int max_steps;
    int current_step;
    int debug_mode; // Set to 1 for standalone, 0 for vectorized

    float reward_approach;
    float reward_complete;
    float reward_grasp;
    float reward_place;
    float penalty_crash;
    float penalty_no_progress;
    float penalty_time;

    float dt;
    float gravity;
    float max_velocity;
    float max_angular_velocity;
    float grip_distance;
    float place_distance;
} DronePickPlace;

void init(DronePickPlace* env) {
    env->drones = calloc(env->num_drones, sizeof(Drone));
    env->objects = calloc(env->num_objects, sizeof(Object));
    env->targets = calloc(env->num_targets, sizeof(TargetZone));
    
    env->dt = 0.02f;
    env->gravity = -9.81f;
    env->max_velocity = 5.0f;
    env->max_angular_velocity = 3.14f;
    env->grip_distance = 0.25f; // todo set sweepable reduction over time for curriculum
    env->place_distance = 0.35f; // todo set sweepable reduction over time for curriculum
}

float randf(float min, float max) {
    return min + (max - min) * ((float)rand() / (float)RAND_MAX);
}

float distance3d(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

void check_and_handle_crashes(DronePickPlace* env) {
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        if (drone->state == STATE_CRASHED) {
            continue;
        }

        if (drone->z < 0.01f) {
            drone->state = STATE_CRASHED;
            drone->z = 0.0f;
            drone->vx = drone->vy = drone->vz = 0.0f;
            drone->wx = drone->wy = drone->wz = 0.0f;
            env->rewards[d] += env->penalty_crash;
        }
    }

    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];

        if (!obj->is_grasped || obj->is_placed) {
            continue;
        }

        if (obj->z < 0.01f) {
            for (int d = 0; d < env->num_drones; d++) {
                Drone* drone = &env->drones[d];

                if (drone->state == STATE_TRANSPORTING && drone->gripper_open < 0.5f) {
                    float dist = distance3d(drone->x, drone->y, drone->z,
                                          obj->x, obj->y, obj->z);
                    if (dist < env->grip_distance) {
                        drone->state = STATE_CRASHED;
                        drone->z = 0.0f;
                        drone->vx = drone->vy = drone->vz = 0.0f;
                        drone->wx = drone->wy = drone->wz = 0.0f;
                        env->rewards[d] += env->penalty_crash;

                        obj->is_grasped = 0;
                        obj->z = 0.1f;
                        obj->vx = obj->vy = obj->vz = 0.0f;

                        break;
                    }
                }
            }
        }
    }
}

bool all_drones_crashed(DronePickPlace* env) {
    for (int d = 0; d < env->num_drones; d++) {
        if (env->drones[d].state != STATE_CRASHED) {
            return false;
        }
    }
    return true;
}

void update_drone_physics(DronePickPlace* env, int drone_idx) {
    Drone* drone = &env->drones[drone_idx];

    if (drone->state == STATE_CRASHED) {
        return;
    }

    int action = env->actions[drone_idx];

    float move_force = 10.0f;
    float rotate_speed = 2.5f;

    switch(action) {
        case MOVE_FORWARD:  // +Y direction (absolute)
            drone->vy += move_force * env->dt;
            break;
        case MOVE_BACKWARD:  // -Y direction (absolute)
            drone->vy -= move_force * env->dt;
            break;
        case MOVE_LEFT:  // -X direction (absolute)
            drone->vx -= move_force * env->dt;
            break;
        case MOVE_RIGHT:  // +X direction (absolute)
            drone->vx += move_force * env->dt;
            break;
        case MOVE_UP:
            drone->vz += move_force * env->dt;
            break;
        case MOVE_DOWN:
            drone->vz -= move_force * env->dt;
            break;
        case ROTATE_LEFT:
            drone->yaw += rotate_speed * env->dt;
            drone->wz = rotate_speed;
            break;
        case ROTATE_RIGHT:
            drone->yaw -= rotate_speed * env->dt;
            drone->wz = -rotate_speed;
            break;
        case GRIPPER_OPEN:
            drone->gripper_open = 1.0f;
            break;
        case GRIPPER_CLOSE:
            drone->gripper_open = 0.0f;
            break;
    }

    float drag = 0.98f;
    drone->vx *= drag;
    drone->vy *= drag;
    drone->vz *= drag;

    drone->vz += env->gravity * env->dt * 0.05f;

    // Clamp velocities
    float speed = sqrtf(drone->vx*drone->vx + drone->vy*drone->vy + drone->vz*drone->vz);
    if (speed > env->max_velocity) {
        float scale = env->max_velocity / speed;
        drone->vx *= scale;
        drone->vy *= scale;
        drone->vz *= scale;
    }

    drone->x += drone->vx * env->dt;
    drone->y += drone->vy * env->dt;
    drone->z += drone->vz * env->dt;

    drone->x = fmaxf(0, fminf(env->world_size, drone->x));
    drone->y = fmaxf(0, fminf(env->world_size, drone->y));
    drone->z = fmaxf(0.05f, fminf(env->max_height, drone->z));

    while (drone->yaw > 3.14159f) drone->yaw -= 2 * 3.14159f; // todo potential bug
    while (drone->yaw < -3.14159f) drone->yaw += 2 * 3.14159f; // todo potential bug

    // Update quaternion from Euler angles (simplified - only yaw for now)
    drone->qw = cosf(drone->yaw / 2.0f);
    drone->qx = 0.0f;
    drone->qy = 0.0f;
    drone->qz = sinf(drone->yaw / 2.0f);

    drone->wx *= 0.9f;
    drone->wy *= 0.9f;
    drone->wz *= 0.9f;
}

void update_grasping(DronePickPlace* env) {
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        
        // Only process if gripper is closed
        if (drone->gripper_open > 0.5f) {
            continue;
        }

        for (int o = 0; o < env->num_objects; o++) {
            Object* obj = &env->objects[o];
            
            // Skip if already grasped by another drone or placed
            if (obj->is_grasped || obj->is_placed) {
                continue;
            }

            float dist = distance3d(drone->x, drone->y, drone->z,
                                   obj->x, obj->y, obj->z);

            if (dist < env->grip_distance * 1.5f) {
                if (dist < env->grip_distance && !obj->is_grasped) {
                    env->stats.grasp_attempts++;
                    obj->is_grasped = 1;
                    drone->state = STATE_TRANSPORTING;
                    env->rewards[d] += env->reward_grasp;
                    env->stats.grasp_successes++;

                    // Only log in debug mode (standalone) to avoid spam from parallel envs
                    if (env->debug_mode) {
                        printf("Drone %d grabbed object %d! (dist=%.2f)\n", d, o, dist);
                    }
                    break;
                }
            }
        }
    }

    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        if (obj->is_grasped && !obj->is_placed) {
            for (int d = 0; d < env->num_drones; d++) {
                Drone* drone = &env->drones[d];
                if (drone->gripper_open < 0.5f) {
                    float dist = distance3d(drone->x, drone->y, drone->z,
                                          obj->x, obj->y, obj->z);
                    if (dist < env->grip_distance * 2) {
                        obj->x = drone->x;
                        obj->y = drone->y;
                        obj->z = drone->z - 0.1f;
                    }
                }
            }
        }
    }
}

void update_placement(DronePickPlace* env) {
    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        
        if (!obj->is_grasped || obj->is_placed) {
            continue;
        }

        for (int t = 0; t < env->num_targets; t++) {
                TargetZone* target = &env->targets[t];
                float dist = distance3d(obj->x, obj->y, obj->z,
                                      target->x, target->y, target->z);
                
                if (dist < env->place_distance * 1.5f) {
                    // Check if any drone released it (attempting placement)
                    // But NOT in first 5 steps to avoid accidental placement at start
                    for (int d = 0; d < env->num_drones; d++) {
                        Drone* drone = &env->drones[d];
                        if (drone->gripper_open > 0.5f && env->current_step > 5) {
                            float drone_dist = distance3d(drone->x, drone->y, drone->z,
                                                         obj->x, obj->y, obj->z);
                            if (drone_dist < env->grip_distance * 2) {
                                
                                if (dist < env->place_distance) {
                                    env->stats.placement_attempts++;
                                    obj->is_placed = 1;
                                    obj->is_grasped = 0;
                                    obj->vx = obj->vy = obj->vz = 0;
                                    target->has_object = 1;
                                    drone->state = STATE_SEARCHING;
                                    env->rewards[d] += env->reward_place;
                                    env->stats.placement_successes++;
                                env->log.perf += 1.0f;
                                env->log.score += 50.0f;
                                
                                // Only log in debug mode (standalone)
                                if (env->debug_mode) {
                                    printf("Drone %d placed object %d in target %d!\n", d, o, t);
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

void compute_observations(DronePickPlace* env) {
    // 45 observations total:
    // Drone state: 14 (pos:3, vel:3, quat:4, ang_vel:3, gripper:1)
    // Objects: 21 (3 objects * 7 features each)
    // Targets: 8 (2 targets * 4 features each)
    // Task info: 2 (time_remaining, task_progress)
    int obs_per_drone = 45;
    
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        int obs_idx = d * obs_per_drone;

        // Drone state (14 values)
        env->observations[obs_idx++] = drone->x / env->world_size;
        env->observations[obs_idx++] = drone->y / env->world_size;
        env->observations[obs_idx++] = drone->z / env->max_height;

        env->observations[obs_idx++] = drone->vx / env->max_velocity;
        env->observations[obs_idx++] = drone->vy / env->max_velocity;
        env->observations[obs_idx++] = drone->vz / env->max_velocity;

        env->observations[obs_idx++] = drone->qw;
        env->observations[obs_idx++] = drone->qx;
        env->observations[obs_idx++] = drone->qy;
        env->observations[obs_idx++] = drone->qz;

        env->observations[obs_idx++] = drone->wx / env->max_angular_velocity;
        env->observations[obs_idx++] = drone->wy / env->max_angular_velocity;
        env->observations[obs_idx++] = drone->wz / env->max_angular_velocity;

        env->observations[obs_idx++] = drone->gripper_open;

        // Object states (7 per object * 3 objects = 21)
        for (int o = 0; o < env->num_objects; o++) {
            Object* obj = &env->objects[o];

            env->observations[obs_idx++] = obj->x / env->world_size;
            env->observations[obs_idx++] = obj->y / env->world_size;
            env->observations[obs_idx++] = obj->z / env->max_height;

            env->observations[obs_idx++] = obj->vx / env->max_velocity;
            env->observations[obs_idx++] = obj->vy / env->max_velocity;
            env->observations[obs_idx++] = obj->vz / env->max_velocity;

            env->observations[obs_idx++] = (float)(obj->is_grasped * 2 + obj->is_placed);
        }

        // Target zone states (4 per target * 2 targets = 8)
        for (int t = 0; t < env->num_targets; t++) {
            TargetZone* target = &env->targets[t];
            // Position (3)
            env->observations[obs_idx++] = target->x / env->world_size;
            env->observations[obs_idx++] = target->y / env->world_size;
            env->observations[obs_idx++] = target->z / env->max_height;
            // Has object (1)
            env->observations[obs_idx++] = (float)target->has_object;
        }

        // Task info (2)
        env->observations[obs_idx++] = 1.0f - (float)env->current_step / env->max_steps; // Time remaining

        // Task progress (ratio of placed objects)
        int placed_count = 0;
        for (int o = 0; o < env->num_objects; o++) {
            if (env->objects[o].is_placed) placed_count++;
        }
        env->observations[obs_idx++] = (float)placed_count / env->num_objects;
    }
}

void add_log(DronePickPlace* env) {
    env->log.episode_length += env->current_step;
    for (int d = 0; d < env->num_drones; d++) {
        env->log.episode_return += env->rewards[d];
    }
    env->log.grasp_success += (env->stats.grasp_successes > 0) ? 1.0f : 0.0f;
    env->log.placement_success += (env->stats.placement_successes > 0) ? 1.0f : 0.0f;
    env->log.n++;
}

bool no_overlap_2d(float x1, float y1, float r1, float x2, float y2, float r2, float extra) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dist2 = dx*dx + dy*dy;
    float min_sep = r1 + r2 + extra;
    return dist2 > (min_sep * min_sep);
};

void c_reset(DronePickPlace* env) {

    env->current_step = 0;

    env->stats.grasp_attempts = 0;
    env->stats.grasp_successes = 0;
    env->stats.placement_attempts = 0;
    env->stats.placement_successes = 0;

    float world = (env->world_size > 0.0f) ? env->world_size : 2.0f;
    float zmax  = (env->max_height > 0.2f) ? env->max_height : 1.0f;

    const float margin = 0.15f;
    const float drone_clearance = 0.20f;
    const float min_obj_radius = 0.08f;
    const float min_tgt_radius = 0.20f;
    const int   max_attempts = 1000;

    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];

        if (obj->radius <= 0.0f) obj->radius = min_obj_radius;

        int attempts = 0;
        while (1) {
            attempts++;
            obj->x = randf(margin + obj->radius, world - margin - obj->radius);
            obj->y = randf(margin + obj->radius, world - margin - obj->radius);
            obj->z = 0.1f;

            int ok = 1;
            for (int k = 0; k < o; k++) {
                Object* other = &env->objects[k];
                if (!no_overlap_2d(obj->x, obj->y, obj->radius, other->x, other->y, other->radius, margin)) {
                    ok = 0; break;
                }
            }
            if (ok || attempts > max_attempts) break;
        }

        obj->vx = obj->vy = obj->vz = 0.0f;
        obj->is_grasped = 0;
        obj->is_placed = 0;
    }

    for (int t = 0; t < env->num_targets; t++) {
        TargetZone* tgt = &env->targets[t];
        if (tgt->radius <= 0.0f) tgt->radius = min_tgt_radius;

        int attempts = 0;
        while (1) {
            attempts++;
            tgt->x = randf(margin + tgt->radius, world - margin - tgt->radius);
            tgt->y = randf(margin + tgt->radius, world - margin - tgt->radius);
            tgt->z = 0.1f;

            int ok = 1;

            for (int o = 0; o < env->num_objects; o++) {
                Object* obj = &env->objects[o];
                if (!no_overlap_2d(tgt->x, tgt->y, tgt->radius, obj->x, obj->y, obj->radius, margin)) {
                    ok = 0; break;
                }
            }
            if (ok) {
                for (int k = 0; k < t; k++) {
                    TargetZone* other = &env->targets[k];
                    if (!no_overlap_2d(tgt->x, tgt->y, tgt->radius, other->x, other->y, other->radius, margin)) {
                        ok = 0; break;
                    }
                }
            }

            if (ok || attempts > max_attempts) break;
        }

        tgt->has_object = 0;
    }

    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];

        int attempts = 0;
        while (1) {
            attempts++;
            drone->x = randf(margin + drone_clearance, world - margin - drone_clearance);
            drone->y = randf(margin + drone_clearance, world - margin - drone_clearance);

            float z_lo = fminf(0.25f, zmax * 0.2f);
            float z_hi = fmaxf(0.6f, zmax * 0.8f);
            z_hi = fminf(z_hi, zmax - 0.05f);
            z_lo = fmaxf(z_lo, 0.15f);
            if (z_lo > z_hi) { z_lo = 0.2f; z_hi = fmaxf(0.4f, zmax * 0.6f); }
            drone->z = randf(z_lo, z_hi);

            int ok = 1;

            for (int o = 0; o < env->num_objects && ok; o++) {
                Object* obj = &env->objects[o];
                if (!no_overlap_2d(drone->x, drone->y, drone_clearance, obj->x, obj->y, obj->radius, margin * 0.5f)) {
                    ok = 0;
                }
            }
            for (int t = 0; t < env->num_targets && ok; t++) {
                TargetZone* tgt = &env->targets[t];
                if (!no_overlap_2d(drone->x, drone->y, drone_clearance, tgt->x, tgt->y, tgt->radius, margin * 0.5f)) {
                    ok = 0;
                }
            }
            for (int k = 0; k < d && ok; k++) {
                Drone* other = &env->drones[k];
                if (!no_overlap_2d(drone->x, drone->y, drone_clearance, other->x, other->y, drone_clearance, margin * 0.5f)) {
                    ok = 0;
                }
            }

            if (ok || attempts > max_attempts) break;
        }

        drone->vx = drone->vy = drone->vz = 0.0f;
        drone->wx = drone->wy = drone->wz = 0.0f;

        drone->yaw = randf(-PI, PI);
        drone->pitch = 0.0f;
        drone->roll = 0.0f;

        // Initialize quaternion from yaw
        drone->qw = cosf(drone->yaw * 0.5f);
        drone->qx = 0.0f;
        drone->qy = 0.0f;
        drone->qz = sinf(drone->yaw * 0.5f);

        drone->gripper_open = 1.0f;
        drone->state = STATE_SEARCHING;
        drone->ticks_without_progress = 0;
    }

    memset(env->rewards, 0, env->num_drones * sizeof(float));
    memset(env->terminals, 0, env->num_drones * sizeof(uint8_t));

    compute_observations(env);
}

void c_step(DronePickPlace* env) {
    env->current_step++;

    memset(env->rewards, 0, env->num_drones * sizeof(float));

    for (int d = 0; d < env->num_drones; d++) {
        update_drone_physics(env, d);

        env->drones[d].ticks_without_progress++;
        if (env->drones[d].ticks_without_progress > 500) {
            env->rewards[d] += env->penalty_no_progress;
        }
    }

    check_and_handle_crashes(env);
    update_grasping(env);
    update_placement(env);

    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        
        if (obj->is_grasped) {
            for (int d = 0; d < env->num_drones; d++) {
                Drone* drone = &env->drones[d];
                if (drone->state == STATE_TRANSPORTING) {
                    obj->x = drone->x;
                    obj->y = drone->y;
                    obj->z = drone->z - 0.15f;
                    obj->vx = drone->vx;
                    obj->vy = drone->vy;
                    obj->vz = drone->vz;
                    break;
                }
            }
        } else if (!obj->is_placed) {
            obj->vz += env->gravity * env->dt;

            // Apply drag
            obj->vx *= 0.98f;
            obj->vy *= 0.98f;
            obj->vz *= 0.98f;

            obj->x += obj->vx * env->dt;
            obj->y += obj->vy * env->dt;
            obj->z += obj->vz * env->dt;

            // Ground collision
            if (obj->z < 0.1f) {
                obj->z = 0.1f;
                obj->vz = 0;
                obj->vx *= 0.8f;  // Friction
                obj->vy *= 0.8f;
            }

            // Boundary constraints
            obj->x = fmaxf(obj->radius, fminf(env->world_size - obj->radius, obj->x));
            obj->y = fmaxf(obj->radius, fminf(env->world_size - obj->radius, obj->y));
        }
    }

    // Compute rewards based on distance to objectives
    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];

        // SPARSE REWARDS - only significant milestones get rewarded
        Object* obj = &env->objects[0];
        TargetZone* target = &env->targets[0];

        if (!obj->is_placed) {
            if (!obj->is_grasped) {
                // PHASE 1: Need to pick up object
                float dist_to_obj = distance3d(drone->x, drone->y, drone->z,
                                              obj->x, obj->y, obj->z);

                float reward = env->penalty_time;
                
                // Small reward only when VERY close and moving toward object
                if (dist_to_obj < 0.3f) {
                    // Check if drone is moving toward object (dot product of velocity and direction)
                    float dx = obj->x - drone->x;
                    float dy = obj->y - drone->y;
                    float dz = obj->z - drone->z;
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    if (dist_sq > 0.0001f) {
                        // Normalize direction
                        float inv_dist = 1.0f / sqrtf(dist_sq);
                        dx *= inv_dist;
                        dy *= inv_dist;
                        dz *= inv_dist;
                        
                        // Dot product with velocity
                        float vel_toward = drone->vx * dx + drone->vy * dy + drone->vz * dz;
                        if (vel_toward > 0.01f) {
                            reward = env->reward_approach;  // Small reward for approaching when close
                        }
                    }
                }
                
                env->rewards[d] += reward;
                
            } else {
                // PHASE 2: Object is grasped - need to transport
                float dist_to_target = distance3d(drone->x, drone->y, drone->z,
                                                 target->x, target->y, target->z);

                float reward = env->penalty_time;

                // Check if drone is moving toward target
                float dx = target->x - drone->x;
                float dy = target->y - drone->y;
                float dz = target->z - drone->z;
                float dist_sq = dx*dx + dy*dy + dz*dz;
                if (dist_sq > 0.0001f) {
                    // Normalize direction
                    float inv_dist = 1.0f / sqrtf(dist_sq);
                    dx *= inv_dist;
                    dy *= inv_dist;
                    dz *= inv_dist;

                    // Dot product with velocity
                    float vel_toward = drone->vx * dx + drone->vy * dy + drone->vz * dz;
                    if (vel_toward > 0.01f) {
                        reward = 2.0f * env->reward_approach; // todo
                    }
                }

                env->rewards[d] += reward;
            }
        } else {
            env->rewards[d] += 0.0f;
        }
    }

    // Check termination conditions
    int all_placed = 1;
    for (int o = 0; o < env->num_objects; o++) {
        if (!env->objects[o].is_placed) {
            all_placed = 0;
            break;
        }
    }

    bool all_crashed = all_drones_crashed(env);

    if (all_placed || all_crashed || env->current_step >= env->max_steps) {
        for (int d = 0; d < env->num_drones; d++) {
            env->terminals[d] = 1;
            if (all_placed) {
                env->rewards[d] += env->reward_complete;
            }
        }
        add_log(env);
        c_reset(env);
    }

    compute_observations(env);
}

void c_render(DronePickPlace* env) {
    if (!IsWindowReady()) {
        InitWindow(800, 600, "Drone Pick & Place Environment");
        SetTargetFPS(30);

        if (env->client == NULL) {
            env->client = calloc(1, sizeof(Client));
            env->client->camera.position = (Vector3){env->world_size * 1.5f, -env->world_size * 0.8f, env->world_size * 1.2f};
            env->client->camera.target = (Vector3){env->world_size/2, env->world_size/2, 0.3f};
            env->client->camera.up = (Vector3){0, 0, 1};
            env->client->camera.fovy = 60.0f;
            env->client->camera.projection = CAMERA_PERSPECTIVE;
        }
    }

    if (IsKeyDown(KEY_ESCAPE)) {
        exit(0);
    }

    BeginDrawing();
    ClearBackground((Color){50, 50, 60, 255});

    BeginMode3D(env->client->camera);

    // Draw ground plane
    DrawCube((Vector3){env->world_size/2, env->world_size/2, -0.01f}, 
             env->world_size, env->world_size, 0.02f, (Color){80, 80, 90, 255});

    // Draw grid lines for reference (make them subtle)
    for (int i = 0; i <= 10; i++) {
        float pos = i * env->world_size / 10.0f;
        DrawLine3D((Vector3){pos, 0, 0}, (Vector3){pos, env->world_size, 0}, (Color){100, 100, 110, 150});
        DrawLine3D((Vector3){0, pos, 0}, (Vector3){env->world_size, pos, 0}, (Color){100, 100, 110, 150});
    }

    for (int d = 0; d < env->num_drones; d++) {
        Drone* drone = &env->drones[d];
        // Drone body color indicates gripper state
        Color drone_color = drone->gripper_open > 0.5f ? (Color){100, 150, 255, 255} : (Color){100, 255, 150, 255};
        DrawCube((Vector3){drone->x, drone->y, drone->z}, 0.15f, 0.15f, 0.08f, drone_color);

        // Draw propellers (simplified)
        DrawCube((Vector3){drone->x, drone->y, drone->z + 0.05f}, 0.25f, 0.02f, 0.01f, (Color){50, 50, 50, 200});
        DrawCube((Vector3){drone->x, drone->y, drone->z + 0.05f}, 0.02f, 0.25f, 0.01f, (Color){50, 50, 50, 200});

        // Draw heading indicator
        Vector3 front = {drone->x + cosf(drone->yaw) * 0.15f,
                        drone->y + sinf(drone->yaw) * 0.15f,
                        drone->z};
        DrawLine3D((Vector3){drone->x, drone->y, drone->z}, front, RED);
    }

    for (int o = 0; o < env->num_objects; o++) {
        Object* obj = &env->objects[o];
        Color obj_color = obj->is_placed ? GREEN : (obj->is_grasped ? YELLOW : RED);
        // Draw cube with size = 2 * radius for each dimension
        float cube_size = obj->radius * 2;
        DrawCube((Vector3){obj->x, obj->y, obj->z}, cube_size, cube_size, cube_size, obj_color);
        // Draw cube wires for better visibility
        DrawCubeWires((Vector3){obj->x, obj->y, obj->z}, cube_size, cube_size, cube_size, BLACK);
    }

    for (int t = 0; t < env->num_targets; t++) {
        TargetZone* target = &env->targets[t];
        // Draw target zones more visibly
        Color zone_color = target->has_object ? (Color){50, 255, 50, 150} : (Color){255, 200, 50, 100};
        DrawCylinderEx((Vector3){target->x, target->y, 0}, 
                       (Vector3){target->x, target->y, 0.03f},
                       target->radius, target->radius, 12, zone_color);
        // Draw target zone outline
        DrawCylinderWiresEx((Vector3){target->x, target->y, 0}, 
                            (Vector3){target->x, target->y, 0.03f},
                            target->radius, target->radius, 12, (Color){255, 255, 255, 200});
    }

    EndMode3D();

    // Draw HUD
    DrawText(TextFormat("Step: %d/%d", env->current_step, env->max_steps), 10, 10, 20, WHITE);
    int placed = 0;
    for (int o = 0; o < env->num_objects; o++) {
        if (env->objects[o].is_placed) placed++;
    }
    DrawText(TextFormat("Placed: %d/%d", placed, env->num_objects), 10, 35, 20, WHITE);

    // Display reward for first drone
    if (env->num_drones > 0) {
        DrawText(TextFormat("Reward: %.3f", env->rewards[0]), 10, 60, 20, YELLOW);

        // Show drone state
        Drone* drone = &env->drones[0];
        const char* state_str = "UNKNOWN";
        switch(drone->state) {
            case STATE_SEARCHING: state_str = "SEARCHING"; break;
            case STATE_APPROACHING: state_str = "APPROACHING"; break;
            case STATE_GRASPING: state_str = "GRASPING"; break;
            case STATE_TRANSPORTING: state_str = "TRANSPORTING"; break;
            case STATE_PLACING: state_str = "PLACING"; break;
            case STATE_CRASHED: state_str = "CRASHED"; break;
        }
        DrawText(TextFormat("State: %s", state_str), 10, 85, 20, SKYBLUE);

        // Show object status
        if (env->num_objects > 0) {
            Object* obj = &env->objects[0];
            DrawText(TextFormat("Object: %s", obj->is_grasped ? "GRASPED" : "FREE"), 10, 110, 20, GREEN);
        }
        DrawText(TextFormat("Z %.3f", drone->z), 10, 135, 20, GREEN);
    }

    EndDrawing();
}

void c_close(DronePickPlace* env) {
    if (env->drones) {
        free(env->drones);
        env->drones = NULL;
    }
    if (env->objects) {
        free(env->objects);
        env->objects = NULL;
    }
    if (env->targets) {
        free(env->targets);
        env->targets = NULL;
    }
    if (env->client) {
        free(env->client);
        env->client = NULL;
    }
    if (IsWindowReady()) {
        CloseWindow();
    }
}
