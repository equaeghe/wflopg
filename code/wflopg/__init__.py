import numpy as np
import xarray as xr
from ruamel.yaml import YAML as yaml

from wflopg import create_turbine
from wflopg import create_wind
from wflopg import layout_geometry
from wflopg import create_wake


class Owflop():
    """The main wind farm layout optimization problem object

    This object stores all the information and data related to a specific
    wind farm layout optimization problem.

    """
    def __init__(self):
        coords = {
            'xy_coord': ['x', 'y'],  # x ~ S→N, y ~ W→E
            'dc_coord': ['d', 'c']   # downwind/crosswind
        }
        # _ds is the main working Dataset
        self._ds = xr.Dataset(coords=coords)
        self._ds.xy_coord.attrs['description'] = (
            "Cartesian coordinates, where ‘x’ corresponds to the "
            "South-North direction and ‘y’ to the West-East direction.")
        self._ds.dc_coord.attrs['description'] = (
            "Cartesian coordinates determined by a given wind direction, "
            "where ‘d’ corresponds to the downwind direction "
            "and ‘c’ to the crosswind direction.")
        # history of layouts as a dict with identifier, layout pairs
        self.layouts = {}

    def load_problem(self, filename):
        """Load wind farm layout optimization problem file

        The file is assumed to be a YAML file in the format described by the
        schema https://bitbucket.org/equaeghe/pseudo_gradients-code/\
        src/master/schemata/wflo_problem-schema.yaml#.

        """
        with open(filename) as f:
            problem = yaml(typ='safe').load(f)
        # extract required parameters directly contained in problem document
        self.problem_uuid = problem['uuid']
        self.turbines = problem['turbines']
        self.turbine_distance = problem.get('turbine_distance', 0)

        # extract and process information and data from linked documents
        with open(problem['turbine']) as f:
            self.process_turbine(yaml(typ='safe').load(f))
        with open(problem['site']) as f:
            self.process_site(
                yaml(typ='safe').load(f),
                self.rotor_radius
            )
        with open(problem['wind_resource']) as f:
            self.process_wind_resource(
                yaml(typ='safe').load(f),
                self.roughness_length,
                problem.get('wind_direction_subdivisions', None),
                problem.get('wind_speeds', None),
                self.cut_in, self.cut_out
            )

        # calculate power information for which no wake calculations are needed
        self.calculate_wakeless_power()

        # process information for string properties
        self.process_wake_model(problem['wake_model'],
                                problem['wake_combination'])
        self.process_objective(problem['objective'])

        # Store downwind and crosswind unit vectors
        self._ds['downwind'] = layout_geometry.generate_downwind(
            self._ds.coords['direction'])
        self._ds['crosswind'] = layout_geometry.generate_crosswind(
            self._ds['downwind'])

        # deal with initial layout
        if 'layout' in problem:
            with open(problem['layout']) as f:
                initial_layout = yaml(typ='safe').load(f)['layout']
        else:
            initial_layout = [[0, 0]]
        self.process_layout(initial_layout)

    def process_turbine(self, turbine):
        self.rotor_radius = turbine['rotor_radius']
        self.hub_height = turbine['hub_height']
        self.rated_power = turbine['rated_power']
        self.rated_speed = turbine['rated_wind_speed']
        self.cut_in = turbine.get('cut_in', 0.0)
        self.cut_out = turbine.get('cut_out', np.inf)
        # define power curve
        if 'power_curve' in turbine:
            pc = np.array(turbine['power_curve'])
            self.power_curve = create_turbine.interpolated_power_curve(
                self.rated_power, self.rated_speed, self.cut_in, self.cut_out,
                pc
            )
        else:
            self.power_curve = create_turbine.cubic_power_curve(
                 self.rated_power, self.rated_speed, self.cut_in, self.cut_out)
        # define thrust curve
        if 'thrust_coefficient' in turbine:
            self.thrust_curve = create_turbine.constant_thrust_curve(
                self.cut_in, self.cut_out, turbine['thrust_coefficient'])
        elif 'thrust_curve' in turbine:
            tc = np.array(turbine['thrust_curve'])
            self.thrust_curve = create_turbine.interpolated_thrust_curve(
                                                 self.cut_in, self.cut_out, tc)
        else:
            raise ValueError("Turbine document should contain either "
                             "a 'thrust_curve' or "
                             "a constant 'thrust_coefficient'")

    def process_site(self, site, rotor_radius):
        self.rotor_constraints = site['rotor_constraints']
        self.site_radius = site['radius'] * 1e3  # km to m
        # TODO: import site parcels and boundaries and together with the rotor
        #       radius create functions to visualize and check the boundary
        #       constraints
        self.roughness_length = site.get('roughness', None)

    def process_wind_resource(self, wind_resource, roughness_length,
                              dir_subs, speeds, cut_in, cut_out):
        self.reference_height = wind_resource['reference_height']
        self.air_density = wind_resource.get('air_density', 1.2041)
        self.atmospheric_stability = wind_resource.get(
            'atmospheric_stability', None)
        self.turbulence_intensity = wind_resource.get(
            'turbulence_intensity', None)

        # Create the wind shear function
        self.wind_shear = create_wind.logarithmic_wind_shear(
                                       self.reference_height, roughness_length)

        wind_rose = wind_resource['wind_rose']

        # Sort wind direction and mass function
        dirs, dir_weights = create_wind.sort_directions(
            wind_rose['directions'], wind_rose['direction_pmf'])

        # Create the conditional wind speed probability mass function
        #
        # NOTE: Of the conditional wind speed probability mass function, only
        #       the values within the [cut_in, cut_out] interval are stored in
        #       self._ds['wind_speed_cpmf'] (defined below), as the others give
        #       no contribution to the power production.
        #
        if 'speed_cweibull' in wind_rose:
            if not speeds:
                raise ValueError(
                    "An array of wind speeds must be specified in case the "
                    "wind resource is formulated in terms of Weibull "
                    "distributions")
            # take wind shear into account
            speeds = self.wind_shear(self.hub_height, np.array(speeds))
            speeds, speed_probs = create_wind.discretize_weibull(
                wind_rose['speed_cweibull'], speeds, cut_in, cut_out)
        elif 'speed_cpmf' in wind_rose and 'speeds' in wind_rose:
            # take wind shear into account
            speeds = self.wind_shear(self.hub_height,
                                     np.array(wind_rose['speeds']))
            speeds, speed_probs = create_wind.conformize_cpmf(
                wind_rose['speed_cpmf'], speeds, cut_in, cut_out)
        else:
            raise ValueError(
                "A conditional wind speed probability distribution "
                "should be given either as parameters for conditional Weibull "
                "distributions or as a conditional probability mass function.")

        # Subdivide wind direction and speed pmfs if needed
        if dir_subs:
            dir_weights, speed_probs = create_wind.subdivide(
                dirs, speeds, dir_weights, speed_probs, dir_subs)
        else:
            dir_weights = xr.DataArray(dir_weights,
                                       coords=[('direction', dirs)])
            speed_probs = xr.DataArray(
                speed_probs,
                coords=[('direction', dirs), ('wind_speed', speeds)]
            )

        # Store pmfs; obtain them from the weight arrays by normalization
        self._ds['direction_pmf'] = dir_weights / dir_weights.sum()
        self._ds['wind_speed_cpmf'] = speed_probs

    def process_layout(self, initial_layout):
        # turbines affected by the wake
        self._ds['layout'] = xr.DataArray(initial_layout,
                                          dims=['target', 'xy_coord'])
        self.layouts['initial'] = xr.DataArray(
            initial_layout,
            dims=['target', 'xy_coord'],
            coords={'xy_coord': self._ds.coords['xy_coord']}
        )
        # turbines causing the wakes
        # NOTE: currently, these are the same as the ones affected
        self._ds['context'] = xr.DataArray(initial_layout,
                                           dims=['source', 'xy_coord'])
        # geometric information (in a general sense)
        self.calculate_geometry()

    def process_wake_model(self, model, combination_rule):
        thrusts = self.thrust_curve(self._ds.coords['wind_speed'])
        # define wake model
        if model == "BPA (IEA37)":
            self.wake_model = create_wake.bpa_iea37(
                thrusts,
                self.rotor_radius,
                self.turbulence_intensity
            )
        elif model == "Jensen":
            self.wake_model = create_wake.jensen(
                thrusts,
                self.rotor_radius,
                self.hub_height,
                self.roughness_length
            )
        elif model == "Jensen according to Frandsen":
            self.wake_model = create_wake.jensen_frandsen(
                thrusts,
                self.rotor_radius,
                self.hub_height,
                self.roughness_length
            )
        elif model == "Jensen with partial wake":
            self.wake_model = create_wake.jensen_averaged(
                thrusts,
                self.rotor_radius,
                self.hub_height,
                self.roughness_length
            )
        elif model == "Jensen according to Frandsen with partial wake":
            self.wake_model = create_wake.jensen_frandsen_averaged(
                thrusts,
                self.rotor_radius,
                self.hub_height,
                self.roughness_length
            )
        else:
            raise ValueError("Unkown wake model specified.")
        # define combination rule
        if combination_rule == "RSS":
            self.combination_rule = create_wake.rss_combination()
        else:
            raise ValueError("Unknown wake combination rule specified.")

    def process_objective(self, objective):
        # we always minimize!
        if objective == "maximize expected power":
            # we minimize the average expected wake loss factor
            self.objective = lambda: self.average_expected_wake_loss_factor
        elif objective == "minimize cost of energy (Mosetti)":
            # we minimize a proxy for the marginal expected cost of energy per
            # turbine
            self.objective = lambda: (
                np.exp(-0.00174 * len(self._ds.coords['target']) ** 2)
                / (1 - self.average_expected_wake_loss_factor)
            )
        else:
            raise ValueError("Unknown objective specified.")

    def calculate_geometry(self):
        # standard coordinates for vectors
        # between all source and target turbines
        self._ds['vector'] = layout_geometry.generate_vector(
                                       self._ds['context'], self._ds['layout'])
        # distances between source and target turbines
        self._ds['distance'] = layout_geometry.generate_distance(
                                                            self._ds['vector'])
        # standard coordinates for unit vectors
        # between all source and target turbines
        self._ds['unit_vector'] = xr.where(
              self._ds['distance'] > 0,
              self._ds['vector'] / self._ds['distance'],
              [0, 0])
        # downwind/crosswind coordinates for vectors
        # between all source and target turbines, for all directions
        self._ds['dc_vector'] = layout_geometry.generate_dc_vector(
            self._ds['vector'], self._ds['downwind'], self._ds['crosswind'])

    def calculate_deficit(self):
        self._ds['deficit'] = self.wake_model(self._ds['dc_vector']
                                              * self.site_radius)
        (self._ds['combined_deficit'],
         self._ds['relative_deficit']) = self.combination_rule(
                                                           self._ds['deficit'])

    def expectation(self, array):
        return (
            array * self._ds['wind_speed_cpmf']
        ).sum(dim='wind_speed').dot(self._ds['direction_pmf'])

    def calculate_wakeless_power(self):
        self._ds['wakeless_power'] = self.power_curve(
            self._ds.coords['wind_speed'])
        self._ds['expected_wakeless_power'] = self.expectation(
            self._ds['wakeless_power'])

    def calculate_power(self):
        # raw values
        power = self.power_curve(
            self._ds.coords['wind_speed'] * (1 - self._ds['combined_deficit'])
        ).transpose('direction', 'wind_speed', 'target')
        wake_loss = self._ds['wakeless_power'] - power
        self._ds['wake_loss_factor'] = (
            wake_loss / self._ds['expected_wakeless_power'])
        # expectations
        self._ds['expected_wake_loss_factor'] = self.expectation(
            self._ds['wake_loss_factor'])
        # turbine average
        self.average_expected_wake_loss_factor = float(
            self._ds['expected_wake_loss_factor'].mean())

    def calculate_relative_wake_loss_vector(self):
        self._ds['relative_wake_loss_vector'] = (
            self._ds['relative_deficit']
            * self._ds['wake_loss_factor']
            * self._ds['unit_vector']
        ).transpose('direction', 'wind_speed', 'source', 'target', 'xy_coord')

    def calculate_push_down_vector(self):
        self._ds['push_down_vector'] = self.expectation(
            self._ds['relative_wake_loss_vector'].sum(dim='source'))

    def calculate_push_back_vector(self):
        self._ds['push_back_vector'] = self.expectation(
            self._ds['relative_wake_loss_vector'].sum(dim='target'))

    def calculate_push_cross_vector(self):
        self._ds['push_cross_vector'] = self.expectation(
            (self._ds['relative_wake_loss_vector'].dot(self._ds['crosswind'])
             * self._ds['crosswind']).sum(dim='source')
        )
