import os, json


def load_file(path):
    print(path, flush=True)
    exists = os.path.exists(path)
    print(exists, flush=True)
    if exists:
        with open(path, 'r') as f:
            print('Opened', flush=True)
            data = json.load(f)
            return  data
    return None

def save_file(features, path):
    with open(path, 'w') as f:
        json.dump(features, f)
    return True

def get_available_techs_for_model(body, avail_fuinction):
    """
    Get a list of available technicians based on a query body.

    Parameters
    ----------
    body : AreaOfCoverRequest
        Query body containing a 'type' key
    avail_fuinction : function
        Function which takes a zone and a date and returns a list of techs available on that date.

    Returns
    -------
    list(str)
        List of technician IDs
    """
    available = avail_fuinction(body.zone, body.date)
    
    if body.type.value == 'bmw':
        available = [ t for t in available if 'SV' in t ]
    elif body.type.value == 'multi':
        available = [ t for t in available if 'MO' in t ]

    return available

def parse_body(body, *args):
    """
    Parse the body of a request to generate a model string.

    Parameters
    ----------
    body : AreaOfCoverRequest
        Request body from client
    body : AreaOfCoverRequest
        Request body from client

    Returns
    -------
    str
        Parsed string for the model
    """
    # Get available technicians
    available_techs = get_available_techs_for_model(body, *args)
    # Parse body
    zone = body.zone.value
    date = body.date
    month = date.month
    weekend_or_BH = date.weekday in [1, 7]
    temp = body.low_temperature
    cluster_type = body.type.value.upper()

    if month in [1,2,3]:
        season = 'WINTER'
    elif month in [4,5,6]:
        season = 'SPRING'
    elif month in [7,8,9]:
        season = 'SUMMER'
    elif month in [10,11,12]:
        sseason = 'FALL'

    if temp:
        temp_str = 'lowC'
    else:
        temp_str = 'highC'

    if weekend_or_BH:
        weekend_str = 'weekend'
    else:
        weekend_str = 'weekday'
    
    zone_prefix = zone.split("  ")[1]
    model_params = f"{zone_prefix}_{cluster_type}_{temp_str}_{weekend_str}_{season}_{len(available_techs)}tech"
    filter_params = dict(
        zone=body.zone.value, 
        cluster_type=cluster_type, 
        low_temperature=temp, 
        weekend=weekend_or_BH, 
        season=season)
    return (model_params, filter_params, available_techs)