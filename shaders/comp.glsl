#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 0, binding = 0, std430) buffer Positions
{
    vec4 pos[];
} positions;

void main()
{
    vec4 current_pos = positions.pos[gl_GlobalInvocationID.x];

    //vec2 velocity = velocities.vel[gl_GlobalInvocationID.x];
    //current_pos += velocity;

    //if (current_pos.x > 0.95 || current_pos.x < -0.95 ||
    //    current_pos.y > 0.95 || current_pos.y < -0.95)
    //{
    //    current_pos = -2.0 * velocity + current_pos * 0.05;
    //}

    positions.pos[gl_GlobalInvocationID.x] = current_pos + vec4(1, 0, 0, 0);
}
