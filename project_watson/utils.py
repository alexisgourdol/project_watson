import time

################
#  DECORATORS  #
################


def simple_time_tracker(method):
    """Time tracker to check the fitting times when training the models."""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if "log_time" in kw:
            name = kw.get("log_name", method.__name__.upper())
            kw["log_time"][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed

def geocoder_here(adress, token=HERE_API_KEY):
    """
    adress: 4 Av du General de Gaulle
     ==>  {'Latitude': 48.85395, 'Longitude': 2.27758}
    """
    geocoderApi = herepy.GeocoderApi(api_key=token)
    res = geocoderApi.free_form(adress)
    res = res.as_dict()
    coords = res["Response"]["View"][0]["Result"][0]["Location"]["DisplayPosition"]
    coords = {k.lower(): v for k, v in coords.items()}
    return coords
