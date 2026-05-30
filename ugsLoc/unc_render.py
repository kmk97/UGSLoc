from einops import reduce

from gaussian_renderer import modified_render_dof

def render_unc(view, gaussians, pipeline, background, hessian_color):
    render_pkg = modified_render_dof(view, gaussians, pipeline, background)
    pred_img = render_pkg["render"]
    pixel_gaussian_counter = render_pkg["pixel_gaussian_counter"]

    render_pkg = modified_render_dof(view, gaussians, pipeline, background, hesssian_color=hessian_color)
    uncertanity_map = reduce(render_pkg["render"], "c h w -> h w", "mean")

    return pred_img, uncertanity_map, pixel_gaussian_counter, render_pkg["depth"]
