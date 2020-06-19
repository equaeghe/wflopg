import sys as _sys
import numpy as _np
import xarray as _xr
import matplotlib.pyplot as _plt
import matplotlib.gridspec as _gs

import wflopg.visualization as vis
from wflopg.create_layout import _take_step, fix_constraints
from wflopg.helpers import rss


def _iterate(step_generator, owflop, max_iterations=_sys.maxsize,
             multiplier=1, scaling=True,
             visualize=False):

    def update_history(owflop, layout, objective, bound=_np.nan,
                       corrections='', multiplier=multiplier):
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = layout
        owflop.history[-1]['objective'] = objective
        owflop.history[-1]['objective_bound'] = bound
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['scale'] = multiplier

    def setup_visualization(owflop):
        axes = {}
        fig = _plt.figure()
        grid = _gs.GridSpec(3, 5)
        axes['windrose'] = fig.add_subplot(grid[0, :2], polar=True)
        vis.draw_windrose(axes['windrose'], owflop._ds.direction_pmf)
        axes['convergence'] = fig.add_subplot(grid[1, :2])
        axes['scaling'] = fig.add_subplot(grid[2, :2],
                                          sharex=axes['convergence'])
        axes['layout'] = fig.add_subplot(grid[:, 2:])
        vis.site_setup(axes['layout'])
        vis.draw_turbines(axes['layout'], owflop, owflop.history[0].layout,
                          proximity=True, in_or_out=True)
        vis.draw_boundaries(axes['layout'], owflop)
        grid.tight_layout(fig)
        _plt.pause(.10)
        return axes

    def iterate_visualization(axes, owflop):
        axes['convergence'].clear()
        vis.draw_convergence(axes['convergence'], owflop.history)
        axes['scaling'].clear()
        vis.draw_scaling(axes['scaling'], owflop.history)
        axes['layout'].clear()
        vis.site_setup(axes['layout'])
        vis.connect_layouts(axes['layout'],
                            [ds.layout for ds in owflop.history])
        vis.draw_turbines(axes['layout'], owflop, owflop.history[0].layout)
        vis.draw_turbines(axes['layout'], owflop, owflop.history[-1].layout,
                          proximity=True, in_or_out=True)
        vis.draw_boundaries(axes['layout'], owflop)
        _plt.pause(0.1)

    if scaling is True:
        scaling = [.8, 1.1]
    elif scaling is False:
        scaling = [1]
    scaler = _xr.DataArray(_np.float64(scaling), dims=('scale',))
    
    owflop.calculate_geometry()
    owflop.calculate_deficit()
    owflop.calculate_power()
    update_history(owflop, owflop._ds.layout, owflop.objective())
    best = start = owflop.history[0].objective
    if visualize:
        axes = setup_visualization(owflop)
    for iteration in range(1, max_iterations+1):
        print(iteration, end=': ')
        owflop._ds['layout'] = owflop.history[-1].layout
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        owflop.calculate_geometry()
        owflop.calculate_deficit()
        # calculate step
        owflop.calculate_relative_wake_loss_vector()
        step = step_generator()
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift # TODO: remove shift before normalization
        step -= step.mean(dim='target')
        # take step
        multiplier = scaler * multiplier
        step = step * multiplier * owflop.rotor_diameter_adim
        _take_step(owflop, step)
        # fix any constraint violation
        corrections = fix_constraints(owflop)
        # evaluate new layout
        owflop.calculate_deficit()
        owflop.calculate_power()
        i = owflop.objective().argmin('scale')
        layout = owflop._ds.layout.isel(scale=i, drop=True)
        current = owflop.objective().isel(scale=i, drop=True)
        bound = best + (start - best) / iteration
        multiplier = multiplier.isel(scale=i, drop=True)
        update_history(owflop, layout, current, bound, corrections, multiplier)
        if visualize:
            iterate_visualization(axes, owflop)
        # check best layout and criteria for early termination
        if current < best:
            best = current
        else:
            if current > bound:
                break
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout, dim='xy'
            ) / owflop.rotor_diameter_adim
            if distance_from_previous.max() < .1:
                break

def pure_away(owflop,
              max_iterations=_sys.maxsize, scaling=[.8, 1.1], multiplier=1,
              visualize=False):
    """Optimize the layout using push-away only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    _iterate(owflop.calculate_push_away_vector, owflop,
             max_iterations, multiplier, scaling,
             visualize)


def pure_back(owflop,
              max_iterations=_sys.maxsize, scaling=[.8, 1.1], multiplier=1,
              visualize=False):
    """Optimize the layout using push-back only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    _iterate(owflop.calculate_push_back_vector, owflop,
             max_iterations, multiplier, scaling,
             visualize)


def mixed_away_and_back(owflop,
                        max_iterations=_sys.maxsize,
                        scaling=[.8, 1.1], multiplier=1,
                        visualize=False):
    """Optimize the layout using a mixture of push-away and push-back

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    def step_generator():
        return (owflop.calculate_push_away_vector()
                + owflop.calculate_push_back_vector()) / 2
    
    _iterate(step_generator, owflop, max_iterations, multiplier, scaling,
             visualize)


def pure_cross(owflop,
               max_iterations=_sys.maxsize, scaling=[.8, 1.1], multiplier=1,
               visualize=False):
    """Optimize the layout using push-cross only

    The problem object owflop is assumed to have a problem loaded, but not
    necessarily have any further actions applied.

    """
    _iterate(owflop.calculate_push_cross_vector, owflop,
             max_iterations, multiplier, scaling,
             visualize)


def multi_adaptive(owflop, max_iterations=_sys.maxsize,
                   scaler=[.5, 1.1], multiplier=3,
                   only_above_average=False, visualize=False):
    if visualize:
        fig = _plt.figure()
        grid = _gs.GridSpec(3, 5)
        ax_windrose = fig.add_subplot(grid[0, :2], polar=True)
        vis.draw_windrose(ax_windrose, owflop._ds.direction_pmf)
        ax_convergence = fig.add_subplot(grid[1, :2])
        ax_scaling = fig.add_subplot(grid[2, :2], sharex=ax_convergence)
        ax_layout = fig.add_subplot(grid[:, 2:])
        vis.site_setup(ax_layout)
        vis.draw_turbines(ax_layout, owflop, owflop._ds.layout,
                          proximity=True, in_or_out=True)
        vis.draw_boundaries(ax_layout, owflop)
        grid.tight_layout(fig)
        _plt.pause(.10)
    scale_coord = ('scale', ['-', '+'])
    method_coord = ('method', ['away', 'back', 'cross'])
    iterations = 0
    best = last = start = 0
    bound = _np.nan
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * _xr.DataArray(
            [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
    )
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    scaler = _xr.DataArray(scaler, coords=[scale_coord])
    scaling = _xr.DataArray([1, 1], coords=[scale_coord])
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            bound = best + (start - best) / iterations
            if last > bound:
                break
        print(iterations, end=': ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        i = objectives.argmin(dim='scale')
        j = objectives.min(dim='scale').argmin('method').item()
        # we continue from best layout
        owflop._ds['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
            * _xr.DataArray(
                [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
        )
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        # update history
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
        )
        owflop.history[-1]['objective'] = (
            objectives.isel(scale=i, drop=True).isel(method=j, drop=True))
        owflop.history[-1]['objective_bound'] = bound
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['method'] = method_coord[1][j]
        owflop.history[-1].attrs['scale'] = (
            scaling.isel(scale=i, drop=True).isel(method=j, drop=True).item())
        if visualize:
            ax_convergence.clear()
            vis.draw_convergence(ax_convergence, owflop.history)
            ax_scaling.clear()
            vis.draw_scaling(ax_scaling, owflop.history)
            ax_layout.clear()
            vis.site_setup(ax_layout)
            vis.connect_layouts(ax_layout,
                                [ds.layout for ds in owflop.history])
            vis.draw_turbines(ax_layout, owflop, owflop.history[0].layout)
            vis.draw_turbines(ax_layout, owflop, owflop.history[-1].layout,
                              proximity=True, in_or_out=True)
            vis.draw_boundaries(ax_layout, owflop)
            _plt.pause(0.1)
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0].objective
        else:
            last = owflop.history[-1].objective
            if last < best:
                best = last
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout,
                dim='xy'
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < owflop.rotor_diameter_adim / 10:
                break
        # calculate new layout
        scaling = scaling.isel(scale=i, drop=True) * scaler
        # first calculate relative_wake_loss_vector just once
        owflop._ds['relative_deficit'] = (
            owflop._ds.relative_deficit.isel(scale=i, drop=True)
                                       .isel(method=j, drop=True))
        owflop._ds['wake_loss_factor'] = (
            owflop._ds.wake_loss_factor.isel(scale=i, drop=True)
                                       .isel(method=j, drop=True))
        owflop._ds['unit_vector'] = (
            owflop._ds.unit_vector.isel(scale=i, drop=True)
                                  .isel(method=j, drop=True))
        owflop.calculate_relative_wake_loss_vector()
        away_step = owflop.calculate_push_away_vector()
        back_step = owflop.calculate_push_back_vector()
        cross_step = owflop.calculate_push_cross_vector()
        # throw steps in one big DataArray
        step = _xr.concat([away_step, back_step, cross_step], 'method')
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        # only take above average steps
        if only_above_average:
            mean_distance = distance.mean(dim='target')
            step *= (distance > mean_distance)
        # generate the different step variants
        step = step * owflop.rotor_diameter_adim * multiplier * scaling
        # take the step
        _take_step(owflop, step)
        corrections = fix_constraints(owflop)
        iterations += 1


def method_chooser(owflop, max_iterations=_sys.maxsize):
    method_coord = ('method', ['away', 'back', 'cross'])
    iterations = 0
    best = last = start = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * _xr.DataArray([1, 1, 1], coords=[method_coord]))
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            bound = best + (start - best) / iterations
            if last > bound:
                break
        print(iterations, end=': ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        j = objectives.argmin('method').values.item()
        # we continue from best layout
        owflop._ds['layout'] = (
            owflop._ds.layout.isel(method=j, drop=True)
            * _xr.DataArray([1, 1, 1], coords=[method_coord])
        )
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        # update history
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(method=j, drop=True))
        owflop.history[-1]['objective'] = objectives.isel(method=j, drop=True)
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['method'] = method_coord[1][j]
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0].objective
        else:
            last = owflop.history[-1].objective
            if last < best:
                best = last
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout,
                dim='xy'
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < owflop.rotor_diameter_adim / 10:
                break
        # calculate new layouts
        # first calculate relative_wake_loss_vector just once
        owflop._ds['relative_deficit'] = (
            owflop._ds.relative_deficit.isel(method=j, drop=True))
        owflop._ds['wake_loss_factor'] = (
            owflop._ds.wake_loss_factor.isel(method=j, drop=True))
        owflop._ds['unit_vector'] = (
            owflop._ds.unit_vector.isel(method=j, drop=True))
        owflop.calculate_relative_wake_loss_vector()
        away_step = owflop.calculate_push_away_vector()
        back_step = owflop.calculate_push_back_vector()
        cross_step = owflop.calculate_push_cross_vector()
        # throw steps in one big DataArray
        step = _xr.concat([away_step, back_step, cross_step], 'method')
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        # take the step, one rotor diameter for the largest pseudo-gradient
        _take_step(owflop, step * owflop.rotor_diameter_adim)
        corrections = fix_constraints(owflop)
        iterations += 1


def multi_wind_resource(owflop, wind_resources, max_iterations=_sys.maxsize,
                        scaler=[.5, 1.1], multiplier=3):
    scale_coord = ('scale', ['-', '+'])
    method_coord = ('method', ['away', 'back', 'cross'])
    # save initial wind resource for objective evaluation
    wind_resource = _xr.Dataset()
    wind_resource['direction_pmf'] = owflop._ds.direction_pmf
    wind_resource['wind_speed_cpmf'] = owflop._ds.wind_speed_cpmf
    iterations = 0
    best = last = start = 0
    corrections = ''
    owflop._ds['layout'] = (
        owflop._ds.layout * _xr.DataArray(
            [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
    )
    owflop._ds['context'] = owflop._ds.layout.rename(target='source')
    owflop.calculate_geometry()
    scaler = _xr.DataArray(scaler, coords=[scale_coord])
    scaling = _xr.DataArray([1, 1], coords=[scale_coord])
    while iterations < max_iterations:
        # stop iterating if no real objective improvement is being made
        if iterations > 0:
            if (
                last - best
                > (start - best) / _np.log2(len(owflop.history) + 2)
            ):
                break
        print(iterations, end=': ')
        owflop.calculate_deficit()
        owflop.calculate_power()
        objectives = owflop.objective()
        i = objectives.argmin(dim='scale')
        j = objectives.min(dim='scale').argmin('method').values.item()
        # we continue from best layout
        owflop._ds['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
            * _xr.DataArray(
                [[1, 1], [1, 1], [1, 1]], coords=[method_coord, scale_coord])
        )
        owflop._ds['context'] = owflop._ds.layout.rename(target='source')
        # update history
        owflop.history.append(_xr.Dataset())
        owflop.history[-1]['layout'] = (
            owflop._ds.layout.isel(scale=i, drop=True)
                             .isel(method=j, drop=True)
        )
        owflop.history[-1]['objective'] = (
            objectives.isel(scale=i, drop=True).isel(method=j, drop=True))
        owflop.history[-1].attrs['corrections'] = corrections
        owflop.history[-1].attrs['method'] = method_coord[1][j]
        owflop.history[-1].attrs['scale'] = (
            scaling.isel(scale=i, drop=True).isel(method=j, drop=True).item())
        if len(owflop.history) == 1:
            best = last = start = owflop.history[0].objective
        else:
            last = owflop.history[-1].objective
            if last < best:
                best = last
            distance_from_previous = rss(
                owflop.history[-1].layout - owflop.history[-2].layout,
                dim='xy'
            )
            # stop iterating if the largest step is smaller than D/10
            if distance_from_previous.max() < owflop.rotor_diameter_adim / 10:
                break
        # calculate new layout
        scaling = scaling.isel(scale=i, drop=True) * scaler
        # swap in driving wind resources
        owflop._ds['direction_pmf'] = wind_resources.direction_pmf
        owflop._ds['wind_speed_cpmf'] = wind_resources.wind_speed_cpmf
        # first calculate relative_wake_loss_vector just once
        owflop._ds['relative_deficit'] = (
            owflop._ds.relative_deficit.isel(scale=i, drop=True)
                                       .isel(method=j, drop=True))
        owflop._ds['wake_loss_factor'] = (
            owflop._ds.wake_loss_factor.isel(scale=i, drop=True)
                                       .isel(method=j, drop=True))
        owflop._ds['unit_vector'] = (
            owflop._ds.unit_vector.isel(scale=i, drop=True)
                                  .isel(method=j, drop=True))
        owflop.calculate_relative_wake_loss_vector()
        away_step = (
            owflop.calculate_push_away_vector().mean(dim='wind_resource'))
        back_step = (
            owflop.calculate_push_back_vector().mean(dim='wind_resource'))
        cross_step = (
            owflop.calculate_push_cross_vector().mean(dim='wind_resource'))
        # throw steps in one big DataArray
        step = _xr.concat([away_step, back_step, cross_step], 'method')
        # normalize the step to the largest pseudo-gradient
        distance = rss(step, dim='xy')
        step /= distance.max('target')
        # remove any global shift
        step -= step.mean(dim='target')
        # generate the different step variants
        step = step * owflop.rotor_diameter_adim * multiplier * scaling
        # take the step
        _take_step(owflop, step)
        corrections = fix_constraints(owflop)
        iterations += 1
        # swap in objective wind resource again
        owflop._ds['direction_pmf'] = wind_resource.direction_pmf
        owflop._ds['wind_speed_cpmf'] = wind_resource.wind_speed_cpmf
