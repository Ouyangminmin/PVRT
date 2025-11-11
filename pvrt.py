# %%
import numpy as np
import pandas as pd
import os
from scipy.integrate import quad
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from datetime import timezone
from datetime import timedelta


# %%
# The following method for calculate the Solar altitude and azimuth is faster than SPA
# method developed by NREL. This method is an improved algorithm of Bourges.


def angle2radian(angle_degrees):
    """
    Convert an angle from degrees to radians.

    Parameters:
        angle_degrees (float or array-like): Input angle in degrees.

    Returns:
        float or array-like: Angle converted to radians.
    """
    return angle_degrees / 180 * np.pi


def radian2angle(angle_radians):
    """
    Convert an angle from radians to degrees.

    Parameters:
        angle_radians (float or array-like): Input angle in radians.

    Returns:
        float or array-like: Angle converted to degrees.
    """
    return angle_radians / np.pi * 180


def hhmmss2hours(hours, minutes, seconds, microseconds=0):
    """
    Convert time components (hours, minutes, seconds, microseconds) to total hours as a float.

    Parameters:
        hours (int): Hours component.
        minutes (int): Minutes component.
        seconds (int): Seconds component.
        microseconds (int, optional): Microseconds component. Defaults to 0.

    Returns:
        float: Total time in hours (e.g., 1 hour 30 minutes returns 1.5).
    """
    total_hours = hours + minutes / 60 + seconds / 3600 + microseconds / 3600000000
    return total_hours


def dts2chips(datetime_index):
    """
    Extract time components from a pandas DatetimeIndex into separate arrays.

    Parameters:
        datetime_index (pandas.DatetimeIndex): Input datetime index.

    Returns:
        list: A list containing arrays of:
            - years
            - months
            - days
            - hours
            - minutes
            - seconds
            - microseconds
            - day of year
            - hour of day (as float, including microseconds)
    """
    years = datetime_index.year.values
    months = datetime_index.month.values
    days = datetime_index.day.values
    hours = datetime_index.hour.values
    minutes = datetime_index.minute.values
    seconds = datetime_index.second.values

    time_components = [
        years,
        months,
        days,
        hours,
        minutes,
        seconds,
    ]
    return time_components


def cal_solar_declination_leap(
    datetime_index, longitude, timezone_longitude, hemisphere="E"
):
    """
    Calculate solar declination, hour angle, and local solar time for dates between 1901-2099.

    Args:
        datetime_index (pandas.DatetimeIndex): Date and time range.
        longitude (float): Longitude of the observation site in degrees.
        timezone_longitude (float): Longitude of the time zone in degrees (e.g., 120 for UTC+8).
        hemisphere (str, optional): Hemisphere indicator, 'E' for East, 'W' for West. Defaults to "E".

    Returns:
        tuple:
            solar_declination (ndarray): Solar declination in degrees.
            hour_angle (ndarray): Hour angle in degrees.
            solar_time (ndarray): Local solar time in hours.
    """
    # Convert datetime to UTC+00 based on timezone longitude and hemisphere
    # Convert datetime to UTC+local  based on timezone longitude and hemisphere
    if hemisphere == "E":
        datetime_utc = datetime_index - timedelta(hours=timezone_longitude / 15)
        datetime_local = datetime_index - timedelta(
            hours=(timezone_longitude - longitude) / 15
        )
    else:
        datetime_utc = datetime_index + timedelta(hours=timezone_longitude / 15)
        datetime_local = datetime_index + timedelta(
            hours=(timezone_longitude - longitude) / 15
        )

    # Extract time components from UTC datetime
    time_comps = dts2chips(datetime_utc)
    years = time_comps[0]
    months = time_comps[1]
    days = time_comps[2]
    hours_val = time_comps[3]  # Hours component
    minutes_val = time_comps[4]  # Minutes component
    seconds_val = time_comps[5]  # Seconds component

    # Angular velocity of Earth's revolution (radians per day)
    angular_velocity = 2 * np.pi / 365.2422

    # Calculate approximate day of year using months and leap year adjustment
    a = years / 4
    b = a - np.floor(a)
    c = np.full_like(years, 32.8)
    c[months <= 2] = 30.6
    c[(b <= 0.005) & (months > 2)] = 31.8
    g = np.floor(30.6 * months - c + 0.5)

    # The revised value for longitude
    if hemisphere == "E":
        longitude_adjustment = -longitude / 15.0
    else:
        longitude_adjustment = longitude / 15.0
    hours_utc = hhmmss2hours(hours_val, minutes_val, seconds_val)

    # Annual accumulated date N
    n = g + days + (hours_utc - longitude_adjustment) / 24

    # Reference day of year N0 for the year 1985
    n0 = 79.6764 + 0.2422 * (years - 1985) - np.floor(0.25 * (years - 1985))

    # Time angle T (radians)
    t = (n - n0) * angular_velocity

    # Solar declination calculation using harmonic series
    solar_declination = (
        0.3723
        + 23.2567 * np.sin(t)
        + 0.1149 * np.sin(2 * t)
        - 0.1712 * np.sin(3 * t)
        - 0.7580 * np.cos(t)
        + 0.3656 * np.cos(2 * t)
        + 0.0201 * np.cos(3 * t)
    )

    # Equation of time (correction for Earth's orbital eccentricity and axial tilt)
    equation_of_time = (
        0.0028
        - 1.9857 * np.sin(t)
        + 9.9059 * np.sin(2 * t)
        - 7.0924 * np.cos(t)
        - 0.6882 * np.cos(2 * t)
    )

    # Solar distance coefficient (calculated but not used in return values)
    solar_distance_coeff = (
        1.000423
        + 0.032359 * np.sin(t)
        + 0.000086 * np.sin(2 * t)
        - 0.008349 * np.cos(t)
        + 0.000115 * np.cos(2 * t)
    )

    # Calculate local true solar time with equation of time correction
    datetime_local_hours = hhmmss2hours(
        datetime_local.hour.values,
        datetime_local.minute.values,
        datetime_local.second.values,
    )
    if hemisphere == "E":
        solar_time = datetime_local_hours + equation_of_time / 60
    else:
        solar_time = datetime_local_hours + equation_of_time / 60

    # Hour angle (degrees from solar noon)
    hour_angle = 15 * (solar_time - 12)

    return solar_declination, hour_angle, solar_time


def cal_solar_angle(latitude, solar_declination, hour_angle):
    """
    Calculate the solar altitude angle and azimuth angle.

    Parameters:
        latitude (float or array-like): Latitude of the observation site in degrees.
        solar_declination (float or array-like): Solar declination in degrees.
        hour_angle (float or array-like): Hour angle in degrees.

    Returns:
        tuple:
            altitude_angle_deg (ndarray): Solar altitude angle in degrees.
            azimuth_angle_deg (ndarray): Solar azimuth angle in degrees.
            altitude_angle_rad (ndarray): Solar altitude angle in radians.
            azimuth_angle_rad (ndarray): Solar azimuth angle in radians.
    """
    # Convert inputs to radians for trigonometric calculations
    latitude_rad = angle2radian(latitude)
    solar_declination_rad = angle2radian(solar_declination)
    hour_angle_rad = angle2radian(hour_angle)

    # Calculate sine of solar altitude angle
    sin_altitude = np.sin(latitude_rad) * np.sin(solar_declination_rad) + np.cos(
        latitude_rad
    ) * np.cos(solar_declination_rad) * np.cos(hour_angle_rad)
    altitude_angle_rad = np.arcsin(sin_altitude)

    # Calculate cosine of solar azimuth angle
    cos_azimuth = (
        np.sin(altitude_angle_rad) * np.sin(latitude_rad)
        - np.sin(solar_declination_rad)
    ) / (np.cos(altitude_angle_rad) * np.cos(latitude_rad))
    azimuth_angle_rad = np.arccos(cos_azimuth)

    # Adjust azimuth angle based on hour angle (before or after solar noon)
    azimuth_angle_rad[hour_angle < 0] = (
        np.pi - azimuth_angle_rad[hour_angle < 0]
    )  # Morning hours
    azimuth_angle_rad[hour_angle >= 0] = (
        np.pi + azimuth_angle_rad[hour_angle >= 0]
    )  # Afternoon hours

    # Convert results back to degrees
    altitude_angle_deg = radian2angle(altitude_angle_rad)
    azimuth_angle_deg = radian2angle(azimuth_angle_rad)

    return altitude_angle_deg, azimuth_angle_deg, altitude_angle_rad, azimuth_angle_rad


def cal_pv_solar_angle(
    datetime_index, timezone_longitude, latitude, longitude, hemisphere="E"
):
    """
    Calculate solar altitude and azimuth angles for a photovoltaic (PV) site.

    Args:
        datetime_index (pandas.DatetimeIndex): Date and time range.
        timezone_longitude (float): Longitude of the time zone in degrees.
        latitude (float): Latitude of the PV site in degrees.
        longitude (float): Longitude of the PV site in degrees.
        hemisphere (str, optional): Hemisphere indicator, 'E' for East, 'W' for West. Defaults to "E".

    Returns:
        tuple:
            altitude_angle_deg (ndarray): Solar altitude angle in degrees.
            azimuth_angle_deg (ndarray): Solar azimuth angle in degrees.
            altitude_angle_rad (ndarray): Solar altitude angle in radians.
            azimuth_angle_rad (ndarray): Solar azimuth angle in radians.
    """
    # Calculate solar declination, hour angle, and solar time
    solar_declination, hour_angle, solar_time = cal_solar_declination_leap(
        datetime_index, longitude, timezone_longitude, hemisphere
    )

    # Calculate solar angles using latitude, declination, and hour angle
    altitude_angle_deg, azimuth_angle_deg, altitude_angle_rad, azimuth_angle_rad = (
        cal_solar_angle(latitude, solar_declination, hour_angle)
    )

    return (
        altitude_angle_deg,
        azimuth_angle_deg,
        altitude_angle_rad,
        azimuth_angle_rad,
    )


# %%
################################################################################
# Class PVRT can calculate the shadow point of PV arrays and $\Omega'_{n,m}$
################################################################################
def func_integrand(xg, yg, zg, xq1, yq1, zq1, lq, wq, hq, kx, ky):
    """Calculate the solid angle in normal direction $\cos \theta' d\Omega'$

    Args:
        xg (float | numpy.ndarray): the location of $P_{g}$ on the X axis.
        yg (float | numpy.ndarray): the location of $P_{g}$ on the Y axis.
        zg (float | numpy.ndarray): the location of $P_{g}$ on the Z axis.
        xq1 (float | numpy.ndarray): the location the lower and left coner of $Q_{n,m}$ on the X axis.
        yq1 (float | numpy.ndarray): the location the lower and left coner of $Q_{n,m}$ on the Y axis.
        zq1 (float | numpy.ndarray): the location the lower and left coner of $Q_{n,m}$ on the Z axis.
        lq (float | numpy.ndarray): the length of $Q_{n,m}$.
        wq (float | numpy.ndarray): the width of $Q_{n,m}$.
        hq (float | numpy.ndarray): the height of $Q_{n,m}$
        kx (float | numpy.ndarray): the coefficients of parametric equation
        ky (float | numpy.ndarray): the coefficients of parametric equation

    Returns:
        (float | numpy.ndarray): $\cos \theta' d\Omega'$
    """
    X = xg - xq1
    Y = yg - yq1
    Z = zg - zq1
    denominator = ((X - kx * lq) ** 2 + (Y - ky * wq) ** 2 + (Z - ky * hq) ** 2) ** 2
    numerator = lq * (Z - ky * hq) * (wq * (Z - ky * hq) - hq * (Y - ky * wq))
    if abs(denominator) < 1e-10:
        return 0.0
    return numerator / denominator


def func_integral(
    args,
    abs_tol=1e-3,
    rel_tol=1e-3,
):
    """ Calculate the solid angle in normal direction $\Omega'_{n,m}$

    Args:
        args (list): The xg, yg, zg, xq1, yq1, zq1, lq, wq, and hq are
            same to the func_integrand. The xg_idx, yg_idx, zg_idx, pv_m, 
            and pv_n are used to mark the $\Omega_{n,m}$. These variables
            are related to the self.pg_xarr_idx, self.pg_yarr_idx, 
            self.pg_zarr_idx, self.ms, self.pv_rows_idx.
        abs_tol (float, optional): Absolute error tolerance. Defaults to 1e-3.
        rel_tol (float, optional): Relative error tolerance. Defaults to 1e-3.

    Returns:
        list: A list containing arrays of:
            [xg_idx, yg_idx, zg_idx, pv_m, pv_n, result]
    """
    (
        xg,
        yg,
        zg,
        xq1,
        yq1,
        zq1,
        lq,
        wq,
        hq,
        xg_idx,
        yg_idx,
        zg_idx,
        pv_m,
        pv_n,
    ) = args

    def inner_integral(ky):
        result, _ = quad(
            lambda kx: func_integrand(xg, yg, zg, xq1, yq1, zq1, lq, wq, hq, kx, ky),
            0,
            1,
            epsabs=abs_tol,
            epsrel=rel_tol,
        )
        return result

    result, _ = quad(inner_integral, 0, 1, epsabs=abs_tol, epsrel=rel_tol)
    return [xg_idx, yg_idx, zg_idx, pv_m, pv_n, result]


def func_integral_batch(batch):
    """Batch processing integration

    Args:
        batch (list): Many args of func_integral compose a list.

    Returns:
        list: Many [xg_idx, yg_idx, zg_idx, pv_m, pv_n, result] compose a list.
    """    
    batch_results = []
    for task in batch:
        try:
            result = func_integral(task)
            batch_results.append(result)
        except Exception as e:
            print(f"Calculate Failed: {task[-5:]}... ERROR: {str(e)}")
            failed_result = task[-5:] + [np.nan]
            batch_results.append(failed_result)
    return batch_results


class PVRT:
    def __init__(
        self,
        pv_lon,
        pv_lat,
        pv_s_xw,
        pv_s_ys,
        pv_s_zb,
        pv_l,
        pv_w,
        pv_h,
        pv_d,
        pv_rows,
        pv_epsilon,
        pg_points=None,
    ):
        """Define the geometric and physical properties of photovoltaic arrays

        Args:
            pv_lon (float): The longitude of PV plant.
            pv_lat (float): The latitude of PV plant.
            pv_s_xw (float): The length of the photovoltaic array on the south
                side of the observation point in the east-west direction
            pv_s_ys (float): The width of the photovoltaic array on the south
                side of the observation point in the south-north direction
            pv_s_zb (float): The height of the photovoltaic array on the south
                side of the observation point in the east-west direction
            pv_l (float): Length of photovolataic array in the south-north
                direction.
            pv_w (float): Width of photovolataic array in the south-north
                direction.
            pv_h (float): Height of photovolataic array in the south-north
                direction.
            pv_d (float): Row spacing of photovoltaic array.
            pv_rows (int): The number of rows of photovoltaic arrays on the 
                south or north side considered in the PVRT model
            pv_epsilon (float): Specific emissivity of photovoltaic array 
                surface
            pg_points (list, optional): A list containing arrays of:
                - pg_xw, pg_xe, pg_xd: The starting point, ending point, and
                    step length of the observation point on the X axis. 
                    xw: x-west; xe: x-east; xd: x-distance.
                - pg_ys, pg_yn, pg_yd: The starting point, ending point, and
                    step length of the observation point on the Y axis.
                    ys: y-south; yn: y_north; yd: y-distance.
                - pg_zb, pg_zt, pg_zd: The starting point, ending point, and
                    step length of the observation point on the Z axis.
                    zb: z-bottom; zt: z-top; z-d: z-distance.
                Defaults to None.
        """    
        # PV geometric properties
        self.pv_lon = pv_lon
        self.pv_lat = pv_lat
        self.pv_s = (pv_s_xw, pv_s_ys, pv_s_zb, pv_l, pv_w, pv_h)
        self.pv_n = (pv_s_xw, pv_s_ys + pv_d, pv_s_zb, pv_l, pv_w, pv_h)
        self.pv_d = pv_d
        self.pv_rows = pv_rows
        self.pv_rows_idx = np.arange(pv_rows)
        self.ms = [0, 1]

        # Observation points geometric properties
        if pg_points is None:
            pg_xw = pv_s_xw
            pg_xe = pv_s_xw + pv_l
            pg_xd = 1.0
            pg_ys = pv_s_ys
            pg_yn = pv_s_ys + pv_d
            pg_yd = 0.1
            pg_zb = 0.0
            pg_zt = pv_s_zb
            pg_zd = pv_s_zb
        else:
            pg_xw, pg_xe, pg_xd, pg_ys, pg_yn, pg_yd, pg_zb, pg_zt, pg_zd = pg_points

        self.pg_xw = pg_xw
        self.pg_xe = pg_xe
        self.pg_xd = pg_xd
        self.pg_ys = pg_ys
        self.pg_yn = pg_yn
        self.pg_yd = pg_yd
        self.pg_zb = pg_zb
        self.pg_zt = pg_zt
        self.pg_zd = pg_zd

        self.pg_xarr = np.arange(pg_xw, pg_xe, pg_xd)
        self.pg_xarr_idx = np.arange(len(self.pg_xarr))

        self.pg_yarr = np.arange(pg_ys, pg_yn, pg_yd)
        self.pg_yarr_idx = np.arange(len(self.pg_yarr))

        self.pg_zarr = np.arange(pg_zb, pg_zt, pg_zd)
        self.pg_zarr_idx = np.arange(len(self.pg_zarr))

        # Physical constant
        self.zerot = -273.15
        self.epsilon = pv_epsilon
        self.sigma = 5.67e-8

        # The ID of PV
        self.pv_idx = "pv_" + "_".join(
            [
                str(round(x, 2))
                for x in [
                    pv_s_xw,
                    pv_s_ys,
                    pv_s_zb,
                    pv_l,
                    pv_w,
                    pv_h,
                    pv_d,
                    pv_rows,
                    pg_xw,
                    pg_xe,
                    pg_xd,
                    pg_ys,
                    pg_yn,
                    pg_yd,
                    pg_zb,
                    pg_zt,
                    pg_zd,
                ]
            ]
        )
        self.pth_cur = os.path.dirname(os.path.abspath("__file__"))

    def add_vis_pv(
        self,
    ):
        """The area of each PV array that can be seen at the observation point
        """    
        little_values = 1.0e-4  # 0.1 mm is a very liitle value in PV plant
        q_geo = list()
        d = self.pv_d
        # loop order: ms, pv_rows_idx, pg_zarr_idx, pg_y_arr_idx
        for m in self.ms:
            # PV arrays on the north side of the $P_g$
            if m == 0:
                x1, y1, z1, l, w, h = self.pv_n
            # PV arrays on the sourth side of the $P_g$
            else:
                x1, y1, z1, l, w, h = self.pv_s
            for n in self.pv_rows_idx:
                # the first odd yg, which is under the lower edge of the PV arrays
                pg_yodd_0 = y1 + (-1) ** m * (n - 1) * d
                for i in self.pg_zarr_idx:
                    zg = self.pg_zarr[i]
                    # the second odd yg, which is in the plane of rectangle PV arrays
                    pg_yodd_1 = y1 + (-1) ** m * (n - 1) * d - w * (z1 - zg) / h
                    for j in self.pg_yarr_idx:
                        yg = self.pg_yarr[j]
                        yg_changed = False
                        if (np.abs(yg - pg_yodd_0) < little_values) | (
                            np.abs(yg - pg_yodd_1) < little_values
                        ):
                            yg += little_values
                            yg_changed = True

                        # Intersection of photovoltaic panel and ground
                        pg_ynode = y1 + (-1) ** m * n * d - w * (h + z1 - zg) / h
                        # the front side of $P_{n,m}$ can be seen at the $P_{g}$
                        if (yg - pg_ynode) < 0:
                            vis_pv_direction = "Front"
                        # the back side of $P_{n,m}$ can be seen at the $P_{g}$
                        else:
                            vis_pv_direction = "Back"

                        # the whole region of $P_{0,m}$ can be seen at the $P_{g}$
                        if n == 0:
                            q_geo_temp = [
                                x1,
                                y1,
                                z1,
                                l,
                                w,
                                h,
                                vis_pv_direction,
                                yg_changed,
                            ]
                            q_geo.append(q_geo_temp)
                            continue

                        # calculate the visiable region of $P_{n,m}$ at $P_g$
                        yq = (
                            y1
                            + (-1) ** m * n * d
                            - (-1) ** m
                            * d
                            * w
                            * (z1 - zg)
                            / (w * (z1 - zg) - h * (y1 + (-1) ** m * (n - 1) * d - yg))
                        )
                        zq = z1 - (-1) ** m * d * h * (z1 - zg) / (
                            w * (z1 - zg) - h * (y1 + (-1) ** m * (n - 1) * d - yg)
                        )
                        if (zq > z1) & (zq < z1 + h):
                            q_geo_temp = [
                                x1,
                                y1 + (-1) ** m * n * d,
                                z1,
                                l,
                                yq - y1 - (-1) ** m * n * d,
                                zq - z1,
                                vis_pv_direction,
                                yg_changed,
                            ]
                        else:
                            q_geo_temp = [
                                x1,
                                y1 + (-1) ** m * n * d,
                                z1,
                                l,
                                w,
                                h,
                                vis_pv_direction,
                                yg_changed,
                            ]
                        q_geo.append(q_geo_temp)

        # PV arrays on the north (m=0) or south (m=1) side of the $P_g$
        col = [
            "pv_xw",
            "pv_ys",
            "pv_zb",
            "pv_l",
            "pv_w",
            "pv_h",
            "vis_pv_direct",
            "pg_yarr_add_little_value",
        ]
        self.pv_vis = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                [
                    self.pg_yarr_idx,
                    self.pg_zarr_idx,
                    self.ms,
                    self.pv_rows_idx,
                ],
                names=[
                    "pg_yarr_idx",
                    "pg_zarr_idx",
                    "direction",
                    "pv_rows_idx",
                ],
            ),
            columns=col,
        )
        # loop order: ms, pv_rows_idx, pg_zarr_idx, pg_y_arr_idx
        idx = pd.MultiIndex.from_product(
            [self.ms, self.pv_rows_idx, self.pg_zarr_idx, self.pg_yarr_idx]
        )
        idx = idx.reorder_levels([3, 2, 0, 1])
        self.pv_vis.loc[idx, col] = q_geo
        self.pv_vis.sort_index(level=[0, 1, 2, 3], inplace=True)

    def add_omega(self, pth_omega=None):
        """Calculate $\Omega'_{n,m}$.

        Args:
            pth_omega (str, optional): the path to save $\Omega'_{n,m}$.
            Defaults to None.
        """        
        def cal_omega(fph_omega):
            # prepare the parameters for all multiprocess calculation
            tasks = []
            task_temp = self.pv_vis[
                ["pv_xw", "pv_ys", "pv_zb", "pv_l", "pv_w", "pv_h"]
            ].copy()
            task_temp["pg_yarr"] = [
                self.pg_yarr[x] for x in task_temp.index.get_level_values(0)
            ]
            task_temp["pg_zarr"] = [
                self.pg_zarr[x] for x in task_temp.index.get_level_values(1)
            ]
            task_temp["pg_yarr_idx"] = task_temp.index.get_level_values(0)
            task_temp["pg_zarr_idx"] = task_temp.index.get_level_values(1)
            task_temp["direction"] = task_temp.index.get_level_values(2)
            task_temp["pv_rows_idx"] = task_temp.index.get_level_values(3)
            tasks = []
            cols = [
                "pg_xarr",
                "pg_yarr",
                "pg_zarr",
                "pv_xw",
                "pv_ys",
                "pv_zb",
                "pv_l",
                "pv_w",
                "pv_h",
                "pg_xarr_idx",
                "pg_yarr_idx",
                "pg_zarr_idx",
                "direction",
                "pv_rows_idx",
            ]
            for xg_idx, xg in enumerate(self.pg_xarr):
                task_temp_xg = task_temp.copy()
                task_temp_xg["pg_xarr"] = xg
                task_temp_xg["pg_xarr_idx"] = xg_idx
                task_temp_xg = task_temp_xg[cols]

                tasks += task_temp_xg.values.tolist()
            self.tasks = tasks
            total_tasks = len(tasks)
            print(
                f"Start calculate the omega with {mp.cpu_count()} cpu cores, the number of tasks are {total_tasks}"
            )

            cols_omega = cols[-5:] + ["omega"]
            with open(fph_omega, "w") as f:
                f.write(",".join(cols_omega) + "\n")

            results = []
            # When the RAM is very small, reducing the chunk_size to a smaller value.
            chunk_size = max(1, total_tasks // (mp.cpu_count() * 4))
            with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
                # submit compute tasks in batches
                futures = {}
                for i in range(0, total_tasks, chunk_size):
                    batch = tasks[i : i + chunk_size]
                    future = executor.submit(func_integral_batch, batch)
                    futures[future] = batch

                # show the process
                pbar = tqdm(total=total_tasks, desc="Calculating", position=0)

                for future in as_completed(futures):
                    batch_results = future.result()
                    results.extend(batch_results)
                    pbar.update(len(batch_results))

                    # write omega to csv file
                    with open(fph_omega, "a") as f:
                        for result in batch_results:
                            f.write(",".join(map(str, result)) + "\n")
                pbar.close()

        # Save or read the omega from csv file.
        if pth_omega is None:
            pth_omega = os.path.dirname(os.path.abspath("__file__"))
        fph_omega = os.path.join(pth_omega, self.pv_idx + "_omega.csv")
        if os.path.exists(fph_omega):
            self.omega = pd.read_csv(fph_omega, index_col=[0, 1, 2, 3, 4])
        else:
            cal_omega(fph_omega)
            self.omega = pd.read_csv(fph_omega, index_col=[0, 1, 2, 3, 4])
        self.omega = np.abs(self.omega)
        self.omega.sort_index(level=[0, 1, 2, 3, 4], kind="stable", inplace=True)

    def add_solar_angle(self, datetime_index, timezone_longitude):
        """Calculate the solar altitude and azimuth.

        Args:
            datetime_index (pandas.core.indexer.datetime.DatetimeIndex): Datetime
            timezone_longitude (float): Timezone . CST is 120.0, UTC-08 is -120.0
        """        
        (
            solar_altitude_deg,
            solar_azimuth_deg,
            solar_altitude_rad,
            solar_azimuth_rad,
        ) = cal_pv_solar_angle(
            datetime_index, timezone_longitude, self.pv_lat, self.pv_lon
        )
        self.pv_solar_angle = pd.DataFrame(
            data=np.stack(
                [
                    solar_altitude_deg,
                    solar_altitude_rad,
                    solar_azimuth_deg,
                    solar_azimuth_rad,
                ]
            ).T,
            index=datetime_index,
            columns=["altitude_deg", "altitude_rad", "azimuth_deg", "azimuth_rad"],
        )
        self.pv_solar_vec = pd.DataFrame(
            data=np.stack(
                [
                    np.cos(solar_altitude_rad) * np.sin(solar_azimuth_rad),
                    np.cos(solar_altitude_rad) * np.cos(solar_azimuth_rad),
                    np.sin(solar_altitude_rad),
                ]
            ).T,
            index=datetime_index,
            columns=["X", "Y", "Z"],
        )
        # drop the nightime solar_altitude and solar azimuth
        idx_drop = self.pv_solar_angle["altitude_deg"] < 0
        self.pv_solar_angle.loc[idx_drop, :] = np.nan
        self.pv_solar_vec.loc[idx_drop, :] = np.nan

    def cal_shadow_points(self, freq="1800s"):
        """Find the locations of the four corners where the photovoltaic array's
            shadow falls on the horizontal plane defined by z=zg.
        Args:
            freq (str, optional): Frequency of datetime_index.
                Defaults to "1800s".
        """        
        # The lower left (west) corner of the PV array as viewed from the front.
        x1, y1, z1, l, w, h = self.pv_s
        # The lower right (east) corner of the PV array as viewed from the front.
        x2, y2, z2 = x1 + l, y1, z1
        # The upper right (east) corner of the PV array as viewed from the front.
        x3, y3, z3 = x1 + l, y1 + w, z1 + h
        # The upper left (west) corner of the PV array as viewed from the front.
        x4, y4, z4 = x1, y1 + w, z1 + h

        def cal_shadow_points(x, y, z, zg, altitude, azimuth):
            """Calculate the projection points of the four vertices of the photovoltaic array on the z=zg

            Args:
                x (float): The X coordinate of corner of the PV array
                y (float): The Y coordinate of corner of the PV array
                z (float): The Z coordinate of corner of the PV array
                zg (float): Horizontal plane defined by z=zg.
                altitude (np.ndarray): Solar altitude angle.
                azimuth (np.ndarray): Solar azimuth angle.

            Returns:
                xs : The X coordinate of corner of the shadow point.
                ys : The Y coordinate of corner of the shadow point.
                zs : The Z coordinate of corner of the shadow point.
            """
            xs = x + (zg - z) / np.sin(altitude) * np.cos(altitude) * np.sin(azimuth)
            ys = y + (zg - z) / np.sin(altitude) * np.cos(altitude) * np.cos(azimuth)
            idx_nan = np.isnan(xs)
            zs = np.full_like(altitude, zg)
            zs[idx_nan] = np.nan
            return xs, ys, zs

        altitude = self.pv_solar_angle["altitude_rad"].values
        azimuth = self.pv_solar_angle["azimuth_rad"].values
        datetime_index = self.pv_solar_angle.index
        pv_shadow = list()

        for idx, i in enumerate(self.pg_zarr):
            i_pv_shadow = list()
            for jdx, j in enumerate(
                [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
            ):
                xs, ys, zs = cal_shadow_points(j[0], j[1], j[2], i, altitude, azimuth)
                i_pv_shadow += [xs, ys, zs]
            i_pv_shadow = pd.DataFrame(
                data=np.stack(i_pv_shadow).T,
                index=pd.MultiIndex.from_product(
                    [[idx], datetime_index], names=["pg_zarr_idx", "datetime_idx"]
                ),
                columns=[
                    "X1",
                    "Y1",
                    "Z1",
                    "X2",
                    "Y2",
                    "Z2",
                    "X3",
                    "Y3",
                    "Z3",
                    "X4",
                    "Y4",
                    "Z4",
                ],
            )
            pv_shadow.append(i_pv_shadow)
        self.pv_shadow = pd.concat(pv_shadow, axis=0)

        self.pv_shadow_revised = self.pv_shadow.copy()
        # Adjust the Y-coordinate based on the row spacing of the PV array.
        d = self.pv_d
        d_shadow = np.abs(self.pv_shadow["Y3"] - self.pv_shadow["Y2"])
        # At sunrise and sunset, the surface can not be irradiated by solar direct radiation
        idx_y_larger_d = d_shadow > d
        self.pv_shadow_revised.loc[idx_y_larger_d, :] = np.nan
        # resampling to period to 30 min
        self.pv_shadow_revised = (
            self.pv_shadow_revised.groupby(level=0).resample(freq, level=1).mean()
        )
        # Determine the Y-axis coordinates for the start and end of the shaded and unshaded areas.
        self.pv_shadow_label = pd.DataFrame(
            data=False, index=self.pv_shadow_revised.index, columns=self.pg_yarr_idx
        )
        Y2 = self.pv_shadow_revised["Y2"].values
        Y3 = self.pv_shadow_revised["Y3"].values
        Yend = np.repeat(np.array([12.0]), len(Y2))
        Ystart = np.repeat(np.array([0.0]), len(Y2))

        # Revised the shadow coordinate through row spacing of photovoltaic array
        def change_pv_shadow_revised(time_log, Y2, Y3):
            mask_arr = np.full_like(self.pv_shadow_label.loc[time_log, :], False)
            row, col = mask_arr.shape
            Y2_chs = np.tile(Y2[time_log], [col, 1]).T
            Y3_chs = np.tile(Y3[time_log], [col, 1]).T
            Y_chs = np.tile(self.pg_yarr, [row, 1])
            log_chs = (Y_chs >= Y2_chs) & (Y_chs <= Y3_chs)
            self.pv_shadow_label.loc[time_log, :] = (
                self.pv_shadow_label.loc[time_log, :].values | log_chs
            )

        # There 5 conditions
        # First condition
        time_log = (Y2 < Y3) & (Y2 >= 0) & (Y3 <= d)
        change_pv_shadow_revised(time_log, Y2, Y3)
        # Second condition
        time_log = (Y2 < Y3) & (Y2 > 12)
        change_pv_shadow_revised(time_log, Y2, Yend)
        change_pv_shadow_revised(time_log, Ystart, Y3 - Yend)
        # Third condition
        time_log = (Y2 < Y3) & (Y2 < 0) & (Y3 >= 0)
        change_pv_shadow_revised(time_log, Ystart, Y3)
        change_pv_shadow_revised(time_log, Y2 + Yend, Yend)
        # Fourth condition
        time_log = (Y2 < Y3) & (Y2 < 0) & (Y3 < 0)
        change_pv_shadow_revised(time_log, Y2 + Yend, Y3 + Yend)
        # Fifth condition
        time_log = Y3 < Y2
        change_pv_shadow_revised(time_log, Y3 + Yend, Y2 + Yend)

    def add_irradiance(
        self,
        datetime_index,
        e_lw,
        e_sw,
        e_sd,
        e_sf,
        t_pvf,
        t_pvb,
    ):
        """Add the irradiance data to PVRT

        Args:
            datetime_index (pandas.core.indexes.datetimes.DatetimeIndex): 
                Datetime of irradiance data.
            e_lw (numpy.ndarray): Long-wave irradiance.
            e_sw (numpy.ndarray): Short-wave irradiance.
            e_sd (numpy.ndarray): Solar direct irradiance.
            e_sf (numpy.ndarray): Solar diffuse irradiance.
            t_pvf (numpy.ndarray): Front temperature of photovoltaic array
            t_pvb (numpy.ndarray): Back temperature of photovoltaic array.

        Raises:
            Exception: If the datetime_index of irradiance is dirrerent with
                the self.pv_shadow_label, raise error.
        """    
        self.datetime_index = datetime_index
        self.e_lw, self.b_lw = e_lw, e_lw / np.pi
        self.e_sw = e_sw
        self.e_sd = e_sd
        self.e_sf, self.b_sf = e_sf, e_sf / np.pi
        self.t_pvf, self.b_pvf = (
            t_pvf,
            self.epsilon * self.sigma * (t_pvf - self.zerot) ** 4 / np.pi,
        )
        self.t_pvb, self.b_pvb = (
            t_pvb,
            self.epsilon * self.sigma * (t_pvb - self.zerot) ** 4 / np.pi,
        )

        # Check consistency of datetime_index between irradiance and shadow_revised
        datetime_index_shadow = self.pv_shadow_label.index.get_loc_level(0)[1]
        if (
            (datetime_index[0] != datetime_index[0])
            | (datetime_index_shadow[-1] != datetime_index[-1])
            | (datetime_index.freqstr != datetime_index_shadow.freqstr)
        ):
            raise Exception(
                "The datetime_index between irradiance and shadow_revised is not consistency"
            )

    def cal_received_flux_density(
        self,
    ):
        """Calculate the radiation flux density received by the surface
        """
        f_mn_index = pd.MultiIndex.from_product(
            [
                self.datetime_index,
                self.pg_yarr_idx,
                self.pg_zarr_idx,
                self.ms,
                self.pv_rows_idx,
            ],
            names=[
                "datetime_index",
                "pg_yarr_idx",
                "pg_zarr_idx",
                "direction",
                "pv_rows_idx",
            ],
        )
        f_index = pd.MultiIndex.from_product(
            [
                self.datetime_index,
                self.pg_yarr_idx,
                self.pg_zarr_idx,
            ],
            names=[
                "datetime_index",
                "pg_yarr_idx",
                "pg_zarr_idx",
            ],
        )
        # get the $\Omega'_{n,m}$ at the $x_g=0$
        self.omega_x_center = self.omega.xs(len(self.pg_xarr_idx) // 2, level=0)
        self.omega_x_center_datetime = pd.DataFrame(
            data=np.tile(self.omega_x_center.values, [len(self.datetime_index), 1]),
            index=f_mn_index,
            columns=["omega"],
        )

        # Solar direct flux density
        def cal_f_sd():
            f_sd_arr = np.tile(self.e_sd, [len(self.pg_yarr_idx), 1]).T
            f_sd_arr = np.tile(f_sd_arr, [len(self.pg_zarr_idx), 1])
            self.f_sd = pd.DataFrame(
                data=f_sd_arr,
                index=self.pv_shadow_label.index,
                columns=self.pv_shadow_label.columns,
            )
            self.f_sd.mask(self.pv_shadow_label, 0.0, inplace=True)

        cal_f_sd()

        # Solar scattered flux density and downward longwave radiation flux density of the atmosphere
        def cal_f_sf_and_air(emit, bright):
            bright_arr = np.repeat(
                bright,
                repeats=len(self.pg_yarr_idx)
                * len(self.pg_zarr_idx)
                * len(self.ms)
                * len(self.pv_rows_idx),
            ).reshape(-1, 1)
            emit_arr = np.repeat(
                emit, repeats=len(self.pg_yarr_idx) * len(self.pg_zarr_idx)
            ).reshape(-1, 1)

            flux_shadow_mn = self.omega_x_center_datetime * bright_arr
            flux_shadow = flux_shadow_mn.groupby(level=[0, 1, 2]).sum()

            flux = -1.0 * (flux_shadow - emit_arr)
            flux = flux.unstack(level=1)
            flux.columns = self.pg_yarr_idx
            flux = flux.reorder_levels([1, 0])
            flux.sort_index(level=[0, 1], inplace=True)
            return flux_shadow_mn, flux

        self.f_sf_shadow_mn, self.f_sf = cal_f_sf_and_air(self.e_sf, self.b_sf)
        self.f_air_shadow_mn, self.f_air = cal_f_sf_and_air(self.e_lw, self.b_lw)

        def cal_f_pv():
            b_pvb_arr = np.repeat(
                self.b_pvb,
                repeats=len(self.pg_yarr_idx)
                * len(self.pg_zarr_idx)
                * len(self.ms)
                * len(self.pv_rows_idx),
            ).reshape(-1, 1)
            b_pvf_arr = np.repeat(
                self.b_pvf,
                repeats=len(self.pg_yarr_idx)
                * len(self.pg_zarr_idx)
                * len(self.ms)
                * len(self.pv_rows_idx),
            ).reshape(-1, 1)
            flux_pvb = self.omega_x_center_datetime * b_pvb_arr
            flux_pvf = self.omega_x_center_datetime * b_pvf_arr
            log_bf = np.tile(
                self.pv_vis["vis_pv_direct"].values, len(self.datetime_index)
            )
            log_bf = log_bf == ["Front"]
            flux_pvb.loc[log_bf, :] = 0
            flux_pvf.loc[~log_bf, :] = 0
            self.f_pv_mn = flux_pvb + flux_pvf
            self.f_pv = self.f_pv_mn.groupby(level=[0, 1, 2]).sum()

            self.f_pv = self.f_pv.unstack(level=1)
            self.f_pv.columns = self.pg_yarr_idx
            self.f_pv = self.f_pv.reorder_levels([1, 0])
            self.f_pv.sort_index(level=[0, 1], inplace=True)

        cal_f_pv()

        self.f_sw = self.f_sf + self.f_sd
        self.f_lw = self.f_air + self.f_pv


# %%
if __name__ == "__main__":
    pth_cur = os.path.dirname(os.path.abspath("__file__"))
    # read the data. The data is not the really observed data
    # Therefore, the results are different with the article
    fph_data = os.path.join(pth_cur, "data_virtual.csv")
    data = pd.read_csv(fph_data, index_col=0)
    data.index = pd.DatetimeIndex(data.index, freq="1800s")
    # virtual data contain some fake data
    # For example, Negative values appear in solar scattered radiation data
    data[["DiffSW", "TotalSW", "DirectSW"]] = np.abs(
        data[["DiffSW", "TotalSW", "DirectSW"]]
    )
    # Init the PVRT
    lat = hhmmss2hours(38, 45, 4, 0)
    lon = hhmmss2hours(100, 48, 3, 0)
    pv_s_xw = -650
    # pv_s_xw = -20
    pv_s_ys = 0
    pv_s_zb = 1.35
    pv_l = 1300
    # pv_l = 40
    pv_w = 3.66
    pv_h = 2.66
    pv_d = 12.0
    pv_rows = 10
    pv_epsilon = 0.83
    pv = PVRT(
        lon, lat, pv_s_xw, pv_s_ys, pv_s_zb, pv_l, pv_w, pv_h, pv_d, pv_rows, pv_epsilon
    )
    # get the visiable region of $P_{n,m}$. In article, it was called $Q_{n,m}$
    pv.add_vis_pv()
    # i9 13900H with 5 GB free RAM might cost about 15 minutes
    # If the free memory is lower than 5 GB, the program may crash

    pv.add_omega()
    # calculate the shadow of PV array
    datetime_index = pd.date_range("2024-08-01 00:00", "2025-07-31 23:59", freq="60s")
    timezone_longitude = 120.0
    pv.add_solar_angle(datetime_index, timezone_longitude)
    pv.cal_shadow_points()
    # calculate the surface received flux density
    pv.add_irradiance(
        data.index,
        data["AirLW"].values,
        data["TotalSW"].values,
        data["DirectSW"].values,
        data["DiffSW"].values,
        data["DSPFT"].values,
        data["DSPBT"].values,
    )
    pv.cal_received_flux_density()

    # output the results
    pv.f_sf.to_csv("solar_diffuse_irradiance_received_by_surface.csv")
    pv.f_sd.to_csv("solar_direct_irradiance_received_by_surface.csv")
    pv.f_sw.to_csv("shortwave_irradiance_received_by_surface.csv")
    pv.f_pv.to_csv("PV_array_emit_irradiance_received_by_surface.csv")
    pv.f_air.to_csv("air_emit_irradiance_received_by_surface.csv")
    pv.f_lw.to_csv("longwave_irradiance_received_by_surface.csv")

    # Plot the irradiance received by surface (z_g=0)
    import string
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "Calibri",
            # "font.style": "italic",
            "font.style": "normal",
            "font.size": 8,
            "mathtext.rm": "Calibri",
            "mathtext.it": "Calibri",
        }
    )
    seasons = ["Spring", "Summer", "Fall", "Winter"]
    seasons_idx = [
        list(pd.date_range("2025-03-01 00:00", "2025-05-31 23:30", freq="1800s")),
        list(pd.date_range("2025-06-01 00:00", "2025-07-31 23:30", freq="1800s"))
        + list(pd.date_range("2024-08-01 00:00", "2024-08-31 23:30", freq="1800s")),
        list(pd.date_range("2024-09-01 00:00", "2024-11-30 23:30", freq="1800s")),
        list(pd.date_range("2024-12-01 00:00", "2025-02-28 23:30", freq="1800s")),
    ]
    abc = string.ascii_lowercase

    ylim = [0, 48]
    lw = 1.0
    x = pv.pg_yarr_idx
    xstr = x / 10.0
    y = np.arange(0, 48, 1)
    ystr = [
        x.strftime("%H:%M")
        for x in pd.date_range("2025-08-30 00:00", "2025-08-30 23:30", freq="1800s")
    ]
    xx, yy = np.meshgrid(x, y)

    # plot the average durial changes of observation data
    nrows = 2
    ncols = 2
    figwidth = round(8.3 / 2.54, 3)
    figheight = round(8.3 / 2.54, 3)
    dpi = 1000
    layout = "constrained"

    def plot_flux_density(flux, level, fph_out):
        """Plot the radiation flux density received by surface

        Args:
            flux (pandas.core.frame.DataFrame): Irradiance received by surface.
            level (numpy.ndarray): Levels of Color.
            fph_out (str): Path for Figure.
        """        
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figwidth, figheight),
            dpi=600,
            layout="constrained",
            sharex=True,
            sharey=True,
        )
        z_df = flux.xs(0, level=0)
        for idx, iseson in enumerate(seasons):
            iz = z_df.loc[seasons_idx[idx], :].copy()
            iz_avg = iz.groupby([x.strftime("%H:%M") for x in iz.index]).mean()
            z = iz_avg.values
            irow = idx // ncols
            icol = idx % ncols
            ax = axs[irow, icol]
            ax.set_title("(" + abc[idx] + ") " + seasons[idx])
            cf = ax.contourf(xx, yy, z, levels=level, cmap="jet")
            c = ax.contour(xx, yy, z, level[::5], colors="black", linewidths=lw)
            ax.clabel(c, inline=1)
            ax.set_xticks(x[::20], [str(int(x)) for x in xstr[::20]])
            ax.set_xticks(x[::10], [], minor=True)
            ax.set_yticks(y[::8], ystr[::8])
            ax.set_yticks(y[::2], [], minor=True)
        fig.supxlabel(r"$y_{g}$" + " (m)")
        fig.supylabel(r"Time (Hour)")
        cbar = fig.colorbar(cf, ax=axs, pad=0.025, aspect=40, fraction=0.15)
        cbar.ax.tick_params()
        fig.savefig(fph_out)
        plt.show(fig)
        plt.close(fig)

    level = np.arange(0, 350, 20)
    plot_flux_density(pv.f_pv, level, "lw_pv_received_by_surface.jpg")

    level = np.arange(0, 350, 20)
    plot_flux_density(pv.f_air, level, "lw_air_received_by_surface.jpg")

    level = np.arange(180, 460, 20)
    plot_flux_density(pv.f_lw, level, "lw_received_by_surface.jpg")

    level = np.arange(0, 180, 20)
    plot_flux_density(pv.f_sf, level, "sw_sf_received_by_surface.jpg")

    level = np.arange(0, 850, 50)
    plot_flux_density(pv.f_sd, level, "sw_sd_received_by_surface.jpg")

    level = np.arange(0, 950, 50)
    plot_flux_density(pv.f_sw, level, "sw_received_by_surface.jpg")

# %%
