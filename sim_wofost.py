"""Real-weather-driven WOFOST 8.1 (water + nitrogen limited) simulation for one field-season."""
import datetime as dt
import math
import numpy as np
import pandas as pd
from pcse.base import WeatherDataProvider, WeatherDataContainer, ParameterProvider
from pcse.input import YAMLCropDataProvider
from pcse.util import reference_ET
from pcse.models import Wofost81_NWLP_CWB_CNB, Wofost81_PP
import yaml

CROPDIR = "./data/wofost81"

SOILS = {
    # name: SMW, SMFCF, SM0, RDMSOL (cm)
    "sandy_loam": dict(SMW=0.09, SMFCF=0.23, SM0=0.40, RDMSOL=80.),
    "silt_loam": dict(SMW=0.14, SMFCF=0.34, SM0=0.46, RDMSOL=110.),
    "clay_loam": dict(SMW=0.21, SMFCF=0.41, SM0=0.50, RDMSOL=100.),
}


def sat_vp(t):
    return 0.6108 * math.exp(17.27 * t / (t + 237.3))  # kPa


class FrameWeather(WeatherDataProvider):
    def __init__(self, df, lat, lon, elev):
        WeatherDataProvider.__init__(self)
        self.description = "station"
        for r in df.itertuples():
            day = r.date.date()
            tmin, tmax = float(r.tmin), float(r.tmax)
            if tmin > tmax:
                tmin, tmax = tmax, tmin
            tavg = 0.5 * (tmin + tmax)
            vap = max(0.2, r.rh / 100.0 * sat_vp(tavg)) * 10.0  # hPa
            irrad = float(r.rg) * 1000.0  # kJ m-2 d-1 -> J m-2 d-1
            wind = max(0.3, float(r.wind) / 3.6)  # km/h -> m/s at 10 m
            wind2 = wind * 4.87 / math.log(67.8 * 10 - 5.42)
            e0, es0, et0 = reference_ET(day, lat, elev, tmin, tmax, irrad, vap, wind2, 0.29, 0.49, ETMODEL="PM")
            wdc = WeatherDataContainer(LAT=lat, LON=lon, ELEV=elev, DAY=day, IRRAD=irrad, TMIN=tmin, TMAX=tmax,
                                       VAP=vap, RAIN=min(24.9, max(0.0, float(r.rain) / 10.0)), E0=e0 / 10., ES0=es0 / 10., ET0=et0 / 10.,
                                       WIND=wind2, TEMP=tavg, SNOWDEPTH=0.0)
            self._store_WeatherDataContainer(wdc, day)


_CROPD = None


def crop_provider():
    global _CROPD
    if _CROPD is None:
        _CROPD = YAMLCropDataProvider(fpath=CROPDIR)
    return _CROPD


def agro(crop, variety, sow, n_events, irr_events, end):
    timed = []
    if n_events:
        timed.append({"event_signal": "apply_n", "name": "N", "comment": "",
                      "events_table": [{d: {"N_amount": a, "N_recovery": 0.7}} for d, a in n_events]})
    if irr_events:
        timed.append({"event_signal": "irrigate", "name": "irr", "comment": "",
                      "events_table": [{d: {"amount": a, "efficiency": 0.75}} for d, a in irr_events]})
    a = [{sow - dt.timedelta(days=1): {"CropCalendar": {"crop_name": crop, "variety_name": variety,
                                                         "crop_start_date": sow, "crop_start_type": "sowing",
                                                         "crop_end_date": end, "crop_end_type": "earliest",
                                                         "max_duration": 200},
                                        "TimedEvents": timed or None, "StateEvents": None}}]
    return a


def run_season(wdp, crop, variety, sow, soil, n_events, irr_events, nsoil, wav, potential=False):
    cp = crop_provider()
    cp.set_active_crop(crop, variety)
    s = dict(SOILS[soil])
    soildata = dict(SMW=s["SMW"], SMFCF=s["SMFCF"], SM0=s["SM0"], CRAIRC=0.06, SOPE=10., KSUB=10.,
                    RDMSOL=s["RDMSOL"], K0=10.,
                    NSOILBASE=nsoil, NSOILBASE_FR=0.025, NAVAILI=0.3 * nsoil, BG_N_SUPPLY=0.091)
    site = dict(WAV=wav, IFUNRN=0, NOTINF=0, SSI=0, SSMAX=0, SMLIM=s["SM0"], CO2=410.)
    params = ParameterProvider(cropdata=cp, soildata=soildata, sitedata=site)
    end = sow + dt.timedelta(days=199)
    am = agro(crop, variety, sow, n_events, irr_events, end)
    model = (Wofost81_PP if potential else Wofost81_NWLP_CWB_CNB)(params, wdp, am)
    model.run_till_terminate()
    out = pd.DataFrame(model.get_output())
    return out, cp
