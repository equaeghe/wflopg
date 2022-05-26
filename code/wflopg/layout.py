import numpy as _np
import xarray as _xr

import wflopg.helpers as _hs


class Layout():
    """A wind farm layout object.

    Parameters
    ----------
    filename : str
        The path to a file containing a wind farm layout
        description satisfying the wind farm layout schema.
    kwargs : dict
        A (nested) `dict` containing a (partial) description
        of a wind farm layout according to the wind farm layout
        schema. It can be used both to override elements of
        the description given in `filename` or to define
        a wind farm layout in its entirety.

    Attributes
    ----------
    STATE_TYPES : frozenset
        The set of state types stored for layouts.
        Contains:

        * `'target'`: wake targets, i.e., turbines that contribute to
          power production (usually all the farm's turbines)
        * `'source'`: wake sources, i.e., turbines that generate wakes
          (usually farm turbines and all the surrounding turbines)
        * `'movable'`: turbines that are allowed to be moved
          (usually all the farm turbines, although some may be kept fixed
          at some point in the design)
        * `'context'`: turbines that belong to the surroundings
          and not to the farm that is being optimized
          (these should not be considered for constraint checks)

    Methods
    -------
    get_positions()
        Get positions.
    initialize_relative_positions(pos_from, pos_to)
        Add and initialize relative positions data in object.
    get_relative_positions()
        Get and, as needed, calculate relative positions.
    get_angles()
        Get and, as needed, calculate angles between locations.
    get_distances()
        Get and, as needed, calculate distances between locations.
    get_normed_relative_positions():
        Get and, as needed, calculate normed relative positions.
    set_positions(positions):
        Set the positions of the layout.
    shift_positions(step):
        Shift the positions of the layout by a given step.

    """
    STATE_TYPES = frozenset({'target', 'source', 'movable', 'context'})

    def __init__(self, filename=None, **kwargs):
        """Create a wind farm layout object."""
        # TODO: add reference to actual wind farm layout schema
        layout_dict = {}
        if filename is not None:
            with open(filename) as f:
                layout_dict.update(_hs.yaml_load(f))
        layout_dict.update(kwargs)
        # TODO: check layout_dict with schema using jsonschema

        layout = layout_dict['layout']
        if isinstance(layout, _xr.core.dataset.Dataset):
            ds = layout
        else:
            if isinstance(layout, list):
                layout_array = _np.array(layout)
            elif isinstance(layout, _np.ndarray):
                layout_array = layout
            else:
                raise ValueError(f"Incorrect type of layout: {layout}.")
            ds = _xr.Dataset(coords={'pos': range(len(layout_array))})
            for name, index in {('x', 0), ('y', 1)}:
                ds[name] = ('pos', layout_array[:, index])
        self._state = ds

        # TODO: add other layout_dict properties as attributes to self?

    def get_positions(self):
        """Get positions.

        Returns
        -------
        An `xarray.Dataset` with `x` and `y` variables
        for absolute positions.

        """
        return self._state[['x', 'y']]

    def set_positions(self, positions):
        """Set the positions of the layout.

        Parameters
        ----------
        positions
            An `xarray.Dataset` with `x` and `y` `xarray.DataArray`s
            with a `pos` coordinate array containing a subset of
            positions present in the object's own `layout` attribute.

        """
        try:
            del self._rel  # rel becomes outdated when layout is changed
        except AttributeError:
            pass
        if not self._state.movable.sel(pos=positions.pos).all():
            raise IndexError("You may not change non-movable turbines.")
        for z in {'x', 'y'}:
            self._state[z].loc[dict(pos=positions.pos)] = positions[z]
        # TODO: do this in one go with
        #
        #   self._state = positions.combine_first(self._state)
        #
        #   Wait until resolution of
        #
        #       https://github.com/pydata/xarray/issues/4220

    def shift_positions(self, step):
        """Shift the positions of the layout by a given step.

        Parameters
        ----------
        step
            An `xarray.Dataset` with `x` and `y` `xarray.DataArray`s
            with a `pos` coordinate array containing a subset of
            positions present in the object's own `layout` attribute.

        """
        self.set_positions(self.positions.loc[dict(pos=step.pos)] + step)

    def get_state(self, *args):
        """Get state information.

        Parameters
        ----------
        args
            State identifiers of `'STATE_TYPES'`.

        Returns
        -------
        An `xarray.Dataset` with the state information
        requested as member variables.

        """
        illegal = set(args) - self.STATE_TYPES
        if illegal:
            raise ValueError(f"The arguments {illegal} are illegal types, "
                             f"i.e., not in {self.STATE_TYPES}.")
        undefined = set(args) - set(self._state.keys())
        if undefined:
            raise ValueError(f"The arguments {undefined} are not currently "
                             "defined for this layout. Use the ‘set_state’ "
                             "method for these first.")
        return self._state[list(args)]

    def set_state(self, **kwargs):
        """Set state information.

        Parameters
        ----------
        kwargs
            A mapping from state identifiers of `'STATE_TYPES'`
            to state values. These can be either `bool`
            or `xarray.DataArray` with `bool` values and with
            the same dimensions as the layout's position variables.

        """
        args = kwargs.keys()
        if not self.STATE_TYPES.issuperset(set(args)):
            raise ValueError(f"The arguments given, {args}, includes illegal "
                             f"types, i.e., not in {self.STATE_TYPES}.")
        for state, val in kwargs.items():
            if isinstance(val, _xr.core.dataarray.DataArray):
                da = val
            elif isinstance(val, bool):
                da = _xr.full_like(self._state.pos, val, dtype=bool)
            else:
                raise ValueError("Incorrect type of description "
                                 f"for state ‘{state}’: ‘{val}’.")
            self._state[state] = da

    def initialize_relative_positions(self, pos_from=None, pos_to=None):
        """Add and initialize relative positions data in object.

        Parameters
        ----------
        pos_from
            Boolean array identifying positions
            to calculate relative positions from
            (`'source'` turbines by default).
        pos_to
            Boolean array identifying positions
            to calculate relative positions to
            (`'target'` turbines by default).

        """
        if pos_from is None:
            pos_from = self._state.source
        if pos_to is None:
            pos_to = self._state.target
        ds = self._state
        self._rel = _xr.Dataset(
            coords={'pos_from': ds.pos[pos_from].rename({'pos': 'pos_from'}),
                    'pos_to': ds.pos[pos_to].rename({'pos': 'pos_to'})}
        )

    def _has_rel_check(self):
        """Raise AttributeError if the _rel attribute is not present."""
        if not hasattr(self, '_rel'):
            raise AttributeError(
                "Call the ‘initialize_relative_positions’ method first.")

    def get_relative_positions(self):
        """Get and, as needed, calculate relative positions.

        Returns
        -------
        An `xarray.Dataset` with `x` and `y` variables
        of relative positions, i.e., differences in coordinate
        values between two absolute positions.

        """
        self._has_rel_check()
        if ('x' not in self._rel) or ('y' not in self._rel):
            xy = self.get_positions()
            self._rel.update(xy.sel(pos=self._rel['pos_to'])
                             - xy.sel(pos=self._rel['pos_from']))
        return self._rel[['x', 'y']]

    def get_angles(self):
        """Get and, as needed, calculate angles between positions.

        Returns
        -------
        An `xarray.DataArray` of angles between absolute positions.

        """
        self._has_rel_check()
        if 'angle' not in self._rel:
            relxy = self.get_relative_positions()
            self._rel['angle'] = _np.arctan2(relxy.y, relxy.x)
        return self._rel.angle

    def get_distances(self):
        """Get and, as needed, calculate distances between positions.

        Returns
        -------
        An `xarray.DataArray` of distances between absolute positions.

        """
        self._has_rel_check()
        if 'distance' not in self._rel:
            relxy = self.get_relative_positions()
            self._rel['distance'] = _np.sqrt(
                _np.square(relxy.x) + _np.square(relxy.y))
        return self._rel.distance

    def get_normed_relative_positions(self):
        """Get and, as needed, calculate normed relative positions.

        Returns
        -------
        An `xarray.Dataset` with `x` and `y` variables of
        normed relative positions, i.e., differences in
        coordinate values between two absolute positions
        normalized by the distance between them.

        """
        self._has_rel_check()
        if ('x_normed' not in self._rel) or ('y_normed' not in self._rel):
            relxy = self.get_relative_positions()
            dists = self.get_distances()
            self._rel.update(
                (relxy / dists).rename({'x': 'x_normed', 'y': 'y_normed'}))
        return self._rel[['x_normed', 'y_normed']]

    @classmethod
    def parallelogram(cls, turbines, acceptable, ratio=1, angle=_np.pi/3,
                      shift=None, rotation=None, randomize=False):
        """Create parallelogram layout.

        A parallelogram layout is a flexibly parametrizable regular grid.
        So it can be used to create a wide class of regular layouts.

        The layout is basically defined by the ratio of two parallelogram
        sides attached to the origin and an angle between them.
        The grid created by tiling the plane with parallelograms
        can be rotated and shifted.

        Parameters
        ----------
        turbines : int
            The (positive) number of turbines n needed in the layout.
        acceptable
            A function that maps a `Layout` to a boolean
            `xarray.DataArray` that indicates which turbines
            are to be considered acceptable. Usually this will
            be the turbines inside the site considered, but it
            may also be different. For example, also turbines
            slightly outside the site may be acceptable, because
            they are used to generate border turbines in a
            subsequent processing step.
        ratio : float
            The (positive) ratio ρ between the most clockwise (CW) side and
            most counterclockwise (CCW) side (0<ρ≤1 without loss of
            generality).
        angle : float
            The (positive) angle α in radians between the sides (0<α<π).
        shift : (float, float)
            The shift (s_CW,s_CCW) of the origin of the grid as fractions of
            side lengths (0≤s_CW,s_CCW<1).
            Random initialization if `None`.
        rotation : float
            The rotation φ in radians of the grid (0<φ<π).
            Random initialization if `None`.
        randomize : Boolean
            Whether or not to randomize the shift and rotation if the algorithm
            seems stuck. (Improves convergence.)

        Returns
        -------
        A `Layout` object with the description of the parallelogram layout.

        """
        """Initialize algorithm parameters"""
        iteration = 0
        scale = 1
        separation = 1
        rng = _np.random.default_rng()
        if shift is None:
            shift = rng.random(2)
        if rotation is None:
            rotation = rng.random() * _np.pi
        """Calculation of initial parallelogram edge lengths

        Let d denote the CW edge length of a parallelogram cell.
        Then the CCW edge length is ρ⋅d.
        So each parallelogram cell has an area ρ⋅d²⋅sin(α).
        The site is contained in a disc of surface π (radius 1 by convention).
        Each cells contributes one turbine to the layout, up to border cases.
        So approximately we should have that n⋅ρ⋅d²⋅sin(α)=π, or
        d = √(π/(n⋅ρ⋅sin(α))).

        """
        d = _np.sqrt(_np.pi / (turbines * ratio * _np.sin(angle)))
        rhod = ratio * d
        col_sep = d
        row_sep = rhod * _np.sin(angle)
        row_shift = rhod * _np.cos(angle)
        """Run loop to find working scale, shift, and rotation"""
        while True:
            """Scale grid lengths with current scale estimate"""
            col_sep *= scale
            row_sep *= scale
            row_shift *= scale
            """Calculation of k that determine row and column count (2⋅k+1)"""
            k_CW = _np.ceil(1 / col_sep)
            k_CCW = _np.ceil(1 / row_sep)
            """Define grid row and column indices"""
            grid = _xr.Dataset(coords={'i': _np.arange(-k_CW, k_CW+1),
                                       'j': _np.arange(-k_CCW, k_CCW+1)})
            """Calculation of grid coordinates"""
            grid = grid.assign(
                x=grid.i * col_sep + grid.j * row_shift,
                y=grid.j * row_sep
            )
            """Flatten data structure and create initial layout"""
            layout = cls.Layout(grid.stack(pos=('i', 'j')))
            """Apply grid rotation and grid shift"""
            rot_cos = _np.cos(rotation)
            rot_sin = _np.sin(rotation)
            layout['x'], layout['y'] = (
                layout.x * rot_cos - layout.y * rot_sin + shift[0] * col_sep,
                layout.x * rot_sin + layout.y * rot_cos + shift[1] * row_sep
            )
            """Determine whether layout has the right number of turbines"""
            in_site = acceptable(layout)
            actual_turbines = sum(in_site)
            if actual_turbines == turbines:
                break
            """Update algorithm parameters if the turbine number is off"""
            iteration += 1
            factor = actual_turbines / turbines
            scale *= _np.sqrt(factor)  # scale area linearly
            """Update shift and rotation to improve convergence

            It may happen for some shift and rotation values that the
            algorithm gets stuck looping between the same scale factors that do
            not correspond to a solution. To avoid that, adding some random
            drift to these parameters can help.

            We use a crude heuristic to determine whether the algorithm is
            stuck. Namely, a symptom is that the actual turbine count is close
            to the desired count. So we keep track of a geometric mean
            (‘separation’) of the absolute difference between them, where half
            the weight is put on the current difference and half on the
            previous mean distance. This mean is then checked against a
            threshold that increases with the number of iterations, which makes
            it more likely to add drift for higher iteration numbers, also a
            symptom of being stuck.

            The standard deviation of the drift is inversely proportional to
            the desired turbine count. This is a gut-feeling heuristic.

            """
            separation = _np.sqrt(separation * _np.abs(factor - 1))
            if randomize and separation < _np.tanh(iteration / turbines):
                shift = _np.remainder(
                    shift + rng.normal(scale=1/turbines, size=2), 1)
                rotation = _np.remainder(
                    rotation + rng.normal(scale=_np.pi/turbines), _np.pi)
        return layout[in_site]

    @classmethod
    def hexagonal(cls, turbines, acceptable,
                  shift=None, rotation=None, randomize=False):
        """Create hexagonal layout.

        A hexagonal layout provides a densest packing of discs.
        So it can be used to create a regular layout with
        a relatively large distance between the turbines.

        Parameters
        ----------
        turbines : int
            The (positive) number of turbines n needed in the layout.
        acceptable
            A function that maps a `Layout` to a boolean
            `xarray.DataArray` that indicates which turbines
            are to be considered acceptable. Usually this will
            be the turbines inside the site considered, but it
            may also be different. For example, also turbines
            slightly outside the site may be acceptable, because
            they are used to generate border turbines in a
            subsequent processing step.
        shift : (float, float)
            The shift (s_CW,s_CCW) of the origin of the grid as fractions of
            side lengths (0≤s_CW,s_CCW<1).
            Random if `None`.
        rotation : float
            The rotation φ in radians of the grid (0<φ<π).
            Random if `None`.
        randomize : Boolean
            Whether or not to randomize the shift and rotation if the algorithm
            seems stuck. (Improves convergence.)

        Returns
        -------
        A `Layout` object with the description of the hexagonal layout.

        """
        return cls.parallelogram(turbines, acceptable,
                                 shift=shift, rotation=rotation,
                                 randomize=randomize)

    @classmethod
    def radial(cls, distances, angles, rotation=None):
        """Create radial layout.

        A radial layout provides a natural grid around a central turbine.
        This layout was conceived for use in generating a so-called
        pre-averaged model, which encodes the effect of the central turbine on
        any other turbine in its surrounding. Therefore, a dense grid is
        typical, which can be used for interpolation.

        Parameters
        ----------
        distances : int or list(float)
            If an integer, the number of equidistant radial distances r
            considered. (Because of its use in the pre-averaged model, 0<r≤2.)
            If a list of floats, these floats should give the distances
            desired.
        angles : int or list(float)
            If an integer, the number of equidistant directions θ considered.
            If a list of floats, these floats should give the directions
            desired (in radians).
        rotation : float
            The rotation φ in radians of the grid (0<φ<2⋅π).
            Random if `None`.

        Returns
        -------
        A `Layout` object with the description of the radial layout.

        """
        if isinstance(distances, int):
            distances = _np.linspace(2, 0, distances, endpoint=False)
        if isinstance(angles, int):
            angles = 2 * _np.pi * _np.linspace(0, 1, angles, endpoint=False)
        if rotation is None:
            rng = _np.random.default_rng()
            rotation = 2 * _np.pi * rng.random()
        angles = _np.remainder(angles + rotation, 2 * _np.pi)
        """Create radial grid"""
        grid = _xr.Dataset(coords={'r': distances, 'theta': angles})
        grid = grid.assign(x=grid.r * _np.cos(grid.theta),
                           y=grid.r * _np.sin(grid.theta))
        origin = _xr.Dataset(coords={'r': [0.], 'theta': [_np.nan]})
        origin = origin.assign(x=(('r', 'theta'), [[0.]]),
                               y=(('r', 'theta'), [[0.]]))
        return cls.Layout(_xr.concat([ds.stack(pos=('r', 'theta'))
                                      for ds in [grid, origin]], 'pos'))
