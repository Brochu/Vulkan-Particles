#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0, std430) buffer Positions
{
    vec2 pos[];
} positions;

layout(set = 0, binding = 1, std430) readonly buffer Velocity
{
    vec2 vel[];
} velocities;

void main()
{
    vec2 current_pos = positions.pos[gl_GlobalInvocationID.x];
    vec2 velocity = velocities.vel[gl_GlobalInvocationID.x];
    current_pos += velocity;

    if (current_pos.x > 0.95 || current_pos.x < -0.95 ||
        current_pos.y > 0.95 || current_pos.y < -0.95)
    {
        current_pos = -2.0 * velocity + current_pos * 0.05;
    }

    positions.pos[gl_GlobalInvocationID.x] = current_pos;
}