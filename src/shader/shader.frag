#version 460

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_tex_coords;

layout(location = 0) out vec4 f_color;

layout(binding = 0, set = 0) uniform sampler2D font_texture;

layout(push_constant) uniform PushConstants
{
    vec2 screen_size;
    int output_in_linear_colorspace;
}
push_constants;

// 0-1 linear  from  0-1 sRGB
vec3 linear_from_srgb(vec3 srgb)
{
    bvec3 cutoff = lessThan(srgb, vec3(0.04045));
    vec3 lower = srgb / vec3(12.92);
    vec3 higher = pow((srgb + vec3(0.055) / vec3(1.055)), vec3(2.4));
    return mix(higher, lower, vec3(cutoff));
}

// 0-1 linear  from  0-1 sRGB
vec4 linear_from_srgba(vec4 srgb)
{
    return vec4(linear_from_srgb(srgb.rgb), srgb.a);
}

void main()
{
    // ALL calculations should be done in gamma space, this includes texture * color and blending
    vec4 texture_color = texture(font_texture, v_tex_coords);
    vec4 color = v_color * texture_color;

    // If output_in_linear_colorspace is true, we are rendering into an sRGB image, for which we'll convert to linear
    // color space.
    // **This will break blending** as it will be performed in linear color space instead of sRGB like egui expects.
    if (push_constants.output_in_linear_colorspace == 1)
    {
        color = linear_from_srgba(color);
    }
    f_color = color;
}