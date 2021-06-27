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
    //return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453) - 0.5;
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

void main()
{
    vec4 current_pos = instances.data[gl_GlobalInvocationID.x].pos;
    vec4 current_vel = instances.data[gl_GlobalInvocationID.x].vel;

    // Move particle
    current_pos += current_vel;
    instances.data[gl_GlobalInvocationID.x].pos = current_pos;

    // Update velocity towards center point
    current_vel = mix(-1 * normalize(current_pos), current_vel, 0.995);
    instances.data[gl_GlobalInvocationID.x].vel = current_vel;
}
