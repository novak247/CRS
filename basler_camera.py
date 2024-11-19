#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Pavel Krsek"
__maintainer__ = "Pavel Krsek"
__email__ = "pavel.krsek@cvut.cz"
__copyright__ = "Copyright © 2023 RMP, CIIRC CVUT in Prague\nAll Rights Reserved."
__license__ = "Use for lectures B3B33ROB1"
__version__ = "1.0"
__date__ = "2024/10/30"
__status__ = "Development"
__credits__ = []
__all__ = []

import numpy as np
from dataclasses import dataclass
from pypylon import pylon
from typing import Any


# ==========================================================================
#
#         ***    CAMERA CLASS    ***
#
# ==========================================================================
@dataclass(init=False)
class BaslerCamera:
    """Class represents the basledr camera interface"""

    #: Camera IP address as string
    ip_address: str
    #: Camera gama correction (default is 1.0)
    gamma: float
    #: Camera gain (default is 0.0, switch off)
    gain: float
    #: Camera exposure time [ms]]
    #: (default value 0.0 mean auto exposure on)
    exposure_time: float
    #: Image capture framerate (frames per sec.)
    #: (default value 0.0 mean automatic / maximal)
    frame_rate: float
    #: Time out for obtaining the image from camera [ms]
    grab_timeout: int
    #: List of attributes to be save to config file.
    #: The attributes are exported/ imported as dictionary
    #: by methodss get_as_dict and set_from_dict.
    config_attrs: list[str]

    def __init__(self):
        self.camera: Any = None
        self.converter: Any = None
        self.connected: bool = False
        self.opened: bool = False
        self.ip_address = ""
        self.gamma = 1.0
        self.gain = 0.0
        self.frame_rate = 0
        self.exposure_time = 0.0
        self.grab_timeout = 1000
        self.config_attrs = [
            "ip_address",
            "grab_timeout",
            "exposure_time",
            "frame_rate",
            "gamma",
            "gain"
        ]

    def get_as_dict(self) -> dict[str, Any]:
        """
        The method returns the class parameters as a dictionary.
        This method is used for writing data into the configuration file.

        Returns:
            dict[str, Any]:
                Parameters names and values in the form of a dictionary.
        """
        ret_dict = {}
        for key in self.config_attrs:
            if hasattr(self, key):
                ret_dict[key] = getattr(self, key)
            else:
                raise AttributeError(f"Missing configuration attribute {key}")
        return ret_dict

    def set_from_dict(self, data: dict[str, Any]) -> None:
        """
        The method sets the class parameters from the dictionary.
        This method is used for reading data from the configuration file.

        Args:
            data (dict[str, Any]):
                Parameters names and values in the form of a dictionary.
        """
        for key, value in data.items():
            if (key in self.config_attrs) & hasattr(self, key):
                setattr(self, key, value)

    def set_parameters(self):
        """
        The method sets the camera parameters (in the camera) by values stored
        in the corresponding attributes of this class.
        """
        self.camera.Gamma.SetValue(self.gamma)
        self.camera.GainAuto.SetValue("Off")
        self.camera.Gain.SetValue(int(self.gain))
        if self.exposure_time > 0:
            self.camera.ExposureAuto.SetValue("Off")
            self.camera.ExposureTime.SetValue(self.exposure_time)
        else:
            self.camera.ExposureAuto.SetValue("Continuous")
        if self.frame_rate > 0:
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.camera.AcquisitionFrameRate.SetValue(self.frame_rate)
        else:
            self.camera.AcquisitionFrameRateEnable.SetValue(False)

    def connect_by_ip(self, ip_addr: str = ""):
        """
        The method connects the camera by its IP address.
        """
        self.opened = False
        self.connected = False
        self.camera = None
        if ip_addr != "":
            self.ip_address = ip_addr
        if self.ip_address != "":
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                raise pylon.RuntimeException("No Basler cameras detected.")
            for device in devices:
                if device.GetIpAddress() == self.ip_address:
                    break
            else:
                raise pylon.RuntimeException(
                    f"Camera with IP {self.ip_address} not found."
                )
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            self.camera.MaxNumBuffer = 5
            self.connected = True
        else:
            raise TypeError("IP address is not defined.")

    def connect_by_name(self, name: str = ""):
        """
        The method connects the camera by its user-defined name.
        """
        self.opened = False
        self.connected = False
        self.camera = None
        if name != "":
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            if len(devices) == 0:
                raise pylon.RuntimeException("No Basler cameras detected.")
            for device in devices:
                if device.GetUserDefinedName() == name:
                    break
            else:
                raise pylon.RuntimeException(
                    f"Camera with name {name} not found."
                )
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
            self.camera.MaxNumBuffer = 5
            self.connected = True
        else:
            raise TypeError("Device name is not defined.")

    def open(self):
        """
        The method opens communication with the connected camera.
        """
        if self.connected:
            self.camera.Open()
            self.opened = True
        if self.opened:
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def start(self):
        """
        The method starts image capturing.
        """
        if self.opened and not self.camera.IsGrabbing():
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def get_image(self, time_out: int = 0) -> np.ndarray:
        """
        The method obtains one image from the camera.

        Args:
            time_out (int): Timeout for image obtaining [ms].

        Returns:
            np.ndarray: The obtained image.
        """
        image = np.array([])
        if self.camera.IsGrabbing():
            time_out = time_out or self.grab_timeout
            res = self.camera.RetrieveResult(
                time_out, pylon.TimeoutHandling_ThrowException)
            if res.GrabSucceeded():
                image = self.converter.Convert(res).GetArray()
            res.Release()
        return image

    def grab_image(self, time_out: int = 0) -> np.ndarray:
        """
        A wrapper for get_image that ensures grabbing is started.

        Args:
            time_out (int): Timeout for image obtaining [ms].

        Returns:
            np.ndarray: The obtained image.
        """
        stop_grab = not self.camera.IsGrabbing()
        if stop_grab:
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        image = self.get_image(time_out)
        if stop_grab:
            self.camera.StopGrabbing()
        return image

    def stop(self):
        """
        The method stops image capturing.
        """
        if self.opened and self.camera.IsGrabbing():
            self.camera.StopGrabbing()

    def close(self):
        """
        The method closes communication with the camera.
        """
        if self.opened:
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()
            self.opened = False
            self.camera.Close()
