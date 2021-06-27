#version 450
#extension GL_ARB_separate_shader_objects : enable

struct PerInstanceData
{
    vec4 pos;
    vec4 vel;
};

layout(set = 0, binding = 0, std430) buffer InstanceData
{
    PerInstanceData data[];
} instances;

float rand(vec2 co)
{
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
}

void main()
{
    //vec4 current_pos = positions.pos[gl_GlobalInvocationID.x];
    //float xmovement = rand(current_pos.xy + 1) / 2;
    //float ymovement = rand(current_pos.xy + 5.0) / 2;

    ////vec2 velocity = velocities.vel[gl_GlobalInvocationID.x];
    ////current_pos += velocity;
    //current_pos.x += xmovement;
    //current_pos.y += ymovement;

    //if (current_pos.x > 50.95 || current_pos.x < -50.95 ||
    //    current_pos.y > 50.95 || current_pos.y < -50.95)
    //{
    //    //current_pos = -2.0 * velocity + current_pos * 0.05;
    //    current_pos.x = 0;
    //    current_pos.y = 0;
    //}

    //positions.pos[gl_GlobalInvocationID.x] = current_pos;

    vec4 current_pos = instances.data[gl_GlobalInvocationID.x].pos;
    current_pos += instances.data[gl_GlobalInvocationID.x].vel;

    instances.data[gl_GlobalInvocationID.x].pos = current_pos;
}
