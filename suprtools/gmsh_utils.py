import numbers
from functools import cached_property
from pathlib import Path
from typing import Literal

import gmsh
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from suprtools._typing import StrPath

gmsh.initialize(['-noenv'])


def set_plugin_options(plugin_name: str, **options):
    for opt_name, opt_val in options.items():
        match opt_val:
            case numbers.Real():
                gmsh.plugin.set_number(plugin_name, opt_name, opt_val)
            case str():
                gmsh.plugin.set_string(plugin_name, opt_name, opt_val)
            case _:
                raise TypeError


def gmsh_modphase(view_tag, timestep_re, timestep_im):
    plugin_name = 'ModulusPhase'
    view_idx = gmsh.view.get_index(view_tag)

    set_plugin_options(
        plugin_name,
        View=view_idx,
        RealPart=timestep_re,
        ImaginaryPart=timestep_im,
    )
    return gmsh.plugin.run(plugin_name)


def gmsh_matheval(components, view_tag=-1, timestep=-1, other_view_tag=-1, other_timestep=-1):
    '''
    '''
    plugin_name = 'MathEval'
    components = np.asarray(components)
    if components.shape not in [(), (1,), (3,), (9,), (3, 3)]:
        raise ValueError

    expr_dict = {
        f'Expression{i}': '' for i in range(9)
    }
    for i, component in enumerate(components.flatten()):
        expr_dict[f'Expression{i}'] = component

    view_idx = gmsh.view.get_index(view_tag)

    set_plugin_options(
        plugin_name,
        View=view_idx,
        TimeStep=timestep,
        OtherView=other_view_tag,
        OtherTimeStep=other_timestep,
        **expr_dict,
    )

    return gmsh.plugin.run(plugin_name)


def merge_curlgrad_fields(
        curlpath: StrPath,
        gradpath: StrPath,
        eps: float = 1e-12,
) -> tuple[tuple[int, int], tuple[int, int]]:
    '''
    '''
    # gmsh.clear()
    gmsh_tags_preopen = set(gmsh.view.get_tags())
    gmsh.open(str(curlpath))
    gmsh.open(str(gradpath))
    gmsh_tags_postopen = set(gmsh.view.get_tags())

    assert gmsh_tags_preopen < gmsh_tags_postopen
    new_tags = gmsh_tags_postopen - gmsh_tags_preopen
    assert len(new_tags) == 2
    curl_tag, grad_tag = min(new_tags), max(new_tags)

    merge_exprs = ['v0', 'v1', f'-w0/(x + {eps})']
    re_tag, im_tag = (
        gmsh_matheval(
            merge_exprs,
            view_tag=curl_tag,
            timestep=timestep,
            other_view_tag=grad_tag,
            other_timestep=timestep,
        )
        for timestep in [0, 1]  # 0: real, 1: imag
    )

    return (curl_tag, grad_tag), (re_tag, im_tag)


def parse_tensor_rank(ch):
    rankdict = {
        'S': 0,
        'V': 1,
        'T': 2,
    }
    return rankdict[ch]


def parse_element_vertices(ch):
    vertices_dict = {
        'P': 1,
        'L': 2,
        'T': 3,
    }
    return vertices_dict[ch]


def parse_single_type_list_data(dtype, n_elts, flat_data):
    if not isinstance(dtype, str):
        raise TypeError
    if not len(dtype) == 2:
        raise ValueError
    tensor_rank = parse_tensor_rank(dtype[0])
    verts_per_elt = parse_element_vertices(dtype[1])

    dims = 3

    data_by_elt = flat_data.reshape(n_elts, -1)
    node_coords_per_elt = dims * verts_per_elt

    node_data = np.reshape(
        data_by_elt[:, :node_coords_per_elt],
        (n_elts, dims, verts_per_elt),
    )
    node_data = np.swapaxes(node_data, 1, 2)

    # () for scalars, (3,) for vctrs, (3, 3) for rank-2 tensors
    field_tensor_shape = (dims,) * tensor_rank
    view_data = np.reshape(
        data_by_elt[:, node_coords_per_elt:],
        (n_elts, -1, verts_per_elt) + field_tensor_shape,
    )
    # n_elts, n_timesteps, n_vert_per_elt, *field_tensor_shape

    return node_data, view_data


def parse_gmsh_list_data(data_tuple):
    '''
    data_tuple:
        returned by gmsh.view.get_list_data
    '''
    # datatypes, n_elts_per, data_per: tuple[
    #   Sequence[str], Sequence[numbers.Integral], NDArray[Any],
    # ] = data_tuple
    return [
        parse_single_type_list_data(*single_type_data_tuple)
        for single_type_data_tuple in zip(*data_tuple)
    ]

#         match dtype:
#             case 'SP':
#                 data_by_position = data.reshape(n_elts, -1)
#                 nodes = data_by_position[:, :dims]
#                 step_data = data_by_position[:, dims:]
#                 return nodes, step_data
#             case 'foo':

#             case _:
#                 raise NotImplementedError


def gmsh_integrate(view_tag):
    '''
    Integrate a scalar field.

    Parameters
    ----------
    view_tag: int
        Tag to the view holding the scalar field.
    '''
    view_idx = gmsh.view.get_index(view_tag)
    plugin_name = 'Integrate'
    set_plugin_options(
        plugin_name,
        View=view_idx,
        OverTime=-1,
        Dimension=2,
        Visible=1,
    )
    result_view_tag = gmsh.plugin.run(plugin_name)
    return parse_gmsh_list_data(gmsh.view.get_list_data(result_view_tag))


def extract_elementnode_data(view_tag, step):
    node_tags, nodes_flat, param_coords = gmsh.model.mesh.get_nodes()
    nodes = nodes_flat.reshape(-1, 3)

    # node_tag_sum = node_tags.sum()
    # visited_sum = 0

    _, elt_tags, elt_data, time, n_cmpnts = gmsh.view.get_model_data(view_tag, step)
    sums = np.zeros((len(nodes), n_cmpnts))
    node_degs = np.zeros(len(nodes))

    for elt_tag, single_elt_data in tqdm(zip(elt_tags, elt_data)):
        elt_type, elt_nodes, elt_dim, elt_ent_tag = gmsh.model.mesh.get_element(elt_tag)
        single_elt_data_shaped = single_elt_data.reshape(-1, n_cmpnts)
        for node_tag, node_data in zip(elt_nodes, single_elt_data_shaped):
            node_idx = np.searchsorted(node_tags, node_tag)

            # if np.any(values[node_idx] != 0):
            #     # print(node_tag)
            #     # np.testing.assert_almost_equal(
            #     #     values[node_idx],
            #     #     node_data,
            #     # )
            #     continue
            # else:
            #     visited_sum += node_idx

            sums[node_idx] += node_data
            node_degs[node_idx] += 1

        # if visited_sum == node_tag_sum:
        #     break

    return nodes, sums / node_degs[:, np.newaxis]


# shape: (num_elts, [coords + vals for each step])
# == split ==>
# (num_elts, pts_per_elt, 3); (num_elts, n_timesteps, pts_per_elt, cmpnts_in_field)

def extract_reim_elementnode_data(view_tag, timestep_re, timestep_im):
    nodes, vals_re = extract_elementnode_data(view_tag, timestep_re)
    nodes_im, vals_im = extract_elementnode_data(view_tag, timestep_im)

    np.testing.assert_equal(nodes, nodes_im)
    return nodes, vals_re + 1j * vals_im


def extract_radtan_elementnode_data(
        rad_view_tag, tan_view_tag,
        timestep_re, timestep_im,
        axis_handling: Literal['mask'] | numbers.Real = 'mask',
):
    nodes_rad, vals_rad = extract_reim_elementnode_data(rad_view_tag, timestep_re, timestep_im)
    nodes_tan, vals_tan = extract_reim_elementnode_data(tan_view_tag, timestep_re, timestep_im)
    np.testing.assert_equal(nodes_rad, nodes_tan)

    r_vals = nodes_rad[:, 0]
    match axis_handling:
        case 'mask':
            mask = (r_vals != 0)
            return (
                nodes_rad[mask],
                vals_rad[mask] + vals_tan[mask] * np.array([0, 0, -1]) / r_vals[mask, np.newaxis],
            )
        case float(eps):
            if eps <= 0:
                raise ValueError
            return (
                nodes_rad,
                vals_rad + vals_tan * np.array([0, 0, -1]) / (r_vals + eps)[:, np.newaxis],
            )
        case _:
            raise TypeError


def radial_integral(re_tag: int, im_tag: int) -> float:
    '''
    Give the volume integral of a complex vector field.

    Parameters
    ----------
    re_tag, im_tag: int
        Tags keying in to the real and imaginary parts of the vector field.
    '''
    magnitude_field_scaled_tag = gmsh_matheval(
        '2 * pi * (v0^2 + v1^2 + v2^2 + w0^2 + w1^2 + w2^2) * x',
        view_tag=re_tag,
        timestep=0,
        other_view_tag=im_tag,
        other_timestep=0,
    )
    _, int_result_arr = gmsh_integrate(magnitude_field_scaled_tag)[0]  # zeroth list data type

    return float(int_result_arr[0, 0, 0])  # zeroth element, zeroth timestep, zeroth component


def vecfield_multi_probe_reim(re_tag, im_tag, points):
    re_part = np.array([
        gmsh.view.probe(re_tag, *point, step=0)[0]
        for point in points
    ])
    im_part = np.array([
        gmsh.view.probe(im_tag, *point, step=0)[0]
        for point in points
    ])

    return re_part + 1j * im_part


class CurlGradField:
    '''
    Encapsulation class for the field patterns returned by the quasi-3D
    small_fem simulation code.
    '''

    reim_tags: tuple[int, int]
    '''View tags of the (computed) real and imaginary parts of the field in gmsh'''

    symmetry_factor: int
    '''An integer giving the factor of reduction of the simulation volume
    vs the corresponding physical volume due to using symmetry'''

    nodes: NDArray[np.float_]
    '''(N, 3) array of coordinates for individual nodes'''

    e_field: NDArray[np.complex_]
    '''(Complex) E-field values computed at each node in `self.nodes`'''

    def __init__(
            self,
            path: StrPath,
            curl_file: StrPath = 'eigenModesCurl.msh',
            grad_file: StrPath = 'eigenModesGrad.msh',
            timestep_reim: tuple[int, int] = (0, 1),
            symmetry_factor: int = 1,
    ):
        path = Path(path)
        self.curl_path = path / curl_file
        self.grad_path = path / grad_file
        self.timesteps_reim = timestep_reim

        if not isinstance(symmetry_factor, numbers.Integral):
            raise TypeError
        elif symmetry_factor <= 0:
            raise ValueError
        self.symmetry_factor = int(symmetry_factor)

        self.curlgrad_tags, self.reim_tags = merge_curlgrad_fields(self.curl_path, self.grad_path)

        rad_tag, tan_tag = self.curlgrad_tags
        self.nodes, self.e_field_raw = extract_radtan_elementnode_data(
            rad_tag, tan_tag,
            *self.timesteps_reim,
        )

        # normalized E-field
        self.e_field = self.e_field_raw / np.sqrt(self.volume_integral)

    @cached_property
    def volume_integral(self) -> float:
        '''
        E* E integrated over the entire mode, accounting for symmetry.
        Cached after first computation to avoid multiple costly
        numerical integrations.
        '''
        return radial_integral(*self.reim_tags) * self.symmetry_factor

    def eval_field(self, points, norm: float | None = 1):
        '''
        Evaluates the field at a point, possibly given a total
        normalization of the field intensity integrated over the entire
        simulation volume, accounting for omitted regions by symmetry.

        Parameters
        ----------
        points: array_like, shape (..., 3)
            Points at which to evaluate field.
        norm: float or None, optional
            If a float, results are normalized such that the intensity
            (E* E) integrated over the full volume (accounting for
            regions omitted from the simulation by symmetry) is equal to
            `norm`. (So results are proportional to sqrt(norm).)
            Defaults to 1. If None, return the raw fields with no
            normalization.

        Returns
        -------
        ndarray, shape (..., 3)
            Array with evaluated fields with same shape as `points`.
        '''
        points = np.asarray(points)
        if points.shape[-1] != 3:
            raise ValueError('Points must have shape 3 in final dimension.')

        normalization_factor = 1
        if norm is not None:
            normalization_factor = np.sqrt(self.volume_integral / norm)
        return vecfield_multi_probe_reim(*self.reim_tags, points) / normalization_factor
