# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from pandapower.control.basic_controller import Controller
from pandapower.toolbox import _detect_read_write_flag, write_to_net
from pandapipes.idx_branch import LENGTH, D, AREA, RHO, VINIT
import math, numpy as np
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


class DynamicControl(Controller):
    """
    Class representing a PID time series controller for a specified element and variable.

    """

    def __init__(self, net, ctrl_element, ctrl_variable, ctrl_element_index, pv_max, pv_min, auto=True, integral=0, dt=1,
                 dir_reversed=False, process_variable=None, process_element=None, process_element_index=None, cv_scaler=1,
                 kp=1, ki=5, Ti= 5, Td=0, kd=0, mv_max=100.00, mv_min=20.00,  profile_name=None,
                 data_source=None, scale_factor=1.0, in_service=True, recycle=True, order=-1, level=-1,
                 drop_same_existing_ctrl=False, matching_params=None,
                 initial_run=False, **kwargs):
        # just calling init of the parent
        if matching_params is None:
            matching_params = {"ctrl_element": ctrl_element, "ctrl_variable": ctrl_variable,
                               "ctrl_element_index": ctrl_element_index}
        super().__init__(net, in_service=in_service, recycle=recycle, order=order, level=level,
                         drop_same_existing_ctrl=drop_same_existing_ctrl,
                         matching_params=matching_params, initial_run=initial_run,
                         **kwargs)

        self.__dict__.update(kwargs)
        #self.kwargs = kwargs

        # data source for time series values
        self.data_source = data_source
        # ids of sgens or loads
        self.ctrl_element_index = ctrl_element_index
        # control element type
        self.ctrl_element = ctrl_element
        self.ctrl_values = None

        self.profile_name = profile_name
        self.scale_factor = scale_factor
        self.applied = False
        self.write_flag, self.ctrl_variable = _detect_read_write_flag(net, ctrl_element, ctrl_element_index, ctrl_variable)
        self.set_recycle(net)

        # PID config
        self.Kp = kp
        self.Kc= 1
        self.Ki = ki
        self.Ti = Ti
        self.Td = Td
        self.Kd = kd
        self.MV_max = mv_max
        self.MV_min = mv_min
        self.PV_max = pv_max
        self.PV_min = pv_min
        self.integral = integral
        self.prev_mv = net[ctrl_element].actual_pos.values
        self.prev_act_pos = net[ctrl_element].actual_pos.values
        self.prev_error = 0
        self.dt = dt
        self.dir_reversed = dir_reversed
        self.gain_effective = ((self.MV_max-self.MV_min)/(self.PV_max - self.PV_min)) * self.Kp
        # selected pv value
        self.process_element = process_element
        self.process_variable = process_variable
        self.process_element_index = process_element_index
        self.cv_scaler = cv_scaler
        self.cv = net[self.process_element][self.process_variable].loc[self.process_element_index]
        self.sp = 0
        self.prev_sp = 0
        self.prev_cv = net[self.process_element][self.process_variable].loc[self.process_element_index]
        self.prev2_cv = net[self.process_element][self.process_variable].loc[self.process_element_index]
        self.I = net[ctrl_element].actual_pos.values
        self.D = 0
        self.auto = auto
        self.yold = self.data_source.get_time_step_value(time_step=0,
                                                  profile_name=self.profile_name,
                                                  scale_factor=self.scale_factor)

        super().set_recycle(net)

    def pid_control(self, error_value):
        """
        Computes the 'velocity' or otherwise known as 'differential' form PID controller
        see "A real-Time Approach to Process Control, pg.112
        System does not suffer from anti-windup but if failure occurs, last value output is held.
        Returns MV position
        """
        ### RTA-2-PC #####

        mv = self.prev_mv + self.Kp*(error_value - self.prev_error + (error_value/self.Ti) -
                                    self.Td*((self.cv - 2*self.prev_cv + self.prev2_cv)/self.dt))

        # MV Saturation
        mv = np.clip(mv, self.MV_min, self.MV_max)

        self.prev2_cv = self.prev_cv
        self.prev_cv = self.cv
        self.prev_error = error_value
        self.prev_mv = mv

        return mv

    # TODO: lag function - need to pass step size 'U' here
    def plant_dynamics(self, desired_mv):
        """
        Takes in the desired valve position (MV value) and computes the actual output depending on
        equipment lag parameters.
        Returns Actual valve position
        """
        if hasattr(self, "act_dynamics"):
            typ= self.act_dynamics
        else:
            # default to instantaneous
            return desired_mv

        # linear
        if typ == "l" :
            t_t_open= self.kwargs.pop("t_t_open", 4)
            t_t_close = self.kwargs.pop("t_t_close", 4)
            # TODO: equation for linear
            actual_pos = desired_mv

        # first order
        elif typ == "fo" :


            # http://techteach.no/simview/lowpass_filter/doc/filter_algorithm.pdf
            #desired_mv = 60
            # works but slightly out at 63% - maybe step size?
            a = np.divide(self.dt, np.divide(self.time_const_s,self.dt) + self.dt)
            actual_pos = (1 - a) * self.prev_act_pos + a * desired_mv
            #########
            #e_k = np.divide(desired_mv, self.time_const_s) + np.divide(self.prev_act_pos, self.time_const_s)
            #actual_pos = self.prev_act_pos + e_k * self.dt

            self.prev_act_pos = actual_pos



        # second order
        elif typ == "so" :
            # TODO: equation for second order
            actual_pos = desired_mv

        else:
            # instantaneous - when incorrect option seledted
            actual_pos = desired_mv

        return actual_pos

    def time_step(self, net, time):
        """
        Get the values of the element from data source
        Write to pandapower net by calling write_to_net()
        If ConstControl is used without a data_source, it will reset the controlled values to the initial values,
        preserving the initial net state.
        """
        self.applied = False

        pv = net[self.process_element][self.process_variable].loc[self.process_element_index]
        self.cv= pv * self.cv_scaler
        self.sp = self.data_source.get_time_step_value(time_step=time,
                                                  profile_name=self.profile_name,
                                                  scale_factor=self.scale_factor)

        if self.auto:
            # Di
            # self.values is the set point we wish to make the output
            if not self.dir_reversed:
                # error= SP-PV
                error_value = self.sp - self.cv
            else:
                # error= SP-PV
                error_value = self.cv - self.sp

            # TODO: hysteresis band
            # if error < 0.01 : error = 0
            desired_mv = self.pid_control(error_value.values)

            actual_pos = self.plant_dynamics(desired_mv)

            self.ctrl_values = actual_pos
        else:
            # Write data source directly to controlled variable
            actual_pos = self.plant_dynamics(self.sp)
            self.ctrl_values = actual_pos
            desired_mv = self.sp

        # Write zeta value to the network
        write_to_net(net, self.ctrl_element, self.ctrl_element_index, self.ctrl_variable, self.ctrl_values, self.write_flag)
        # Write the desired MV value to results for future plotting
        write_to_net(net, self.ctrl_element, self.ctrl_element_index, "desired_mv" , desired_mv, self.write_flag)

    def is_converged(self, net):
        """
        Actual implementation of the convergence criteria: If controller is applied, it can stop
        """
        return self.applied

    def control_step(self, net):
        """
        Set applied to True, which means that the values set in time_step have been included in the load flow calculation.
        """
        self.applied = True

    def __str__(self):
        return super().__str__() + " [%s.%s]" % (self.ctrl_element, self.ctrl_variable)
