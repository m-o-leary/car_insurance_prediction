from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, validator, ValidationError
from rest.models.cold_call import Call, Model

import datetime

# Instantiate the router
router = APIRouter( prefix="/insurance_model" )

@router.get("/models/", response_model=List[Model], status_code=200, tags=["Model"])
async def get_model_data(model_id: str):
    """
    Return all available models.

    """
    model_list = []

    return { "models": model_list }

# @router.get("/technicians/types",response_model=List[CaseEnum], status_code=200, tags=["Master Data"])
# async def get_technicians_types():
#     """
#     Return types of technicians.

#     """
#     return [ case.value for case in CaseEnum ]

# @router.get("/zones/", response_model=ZoneList,  status_code=200, tags=["Master Data"])
# async def get_zone_ids(zones: List = Depends(get_zone_list)):
#     """
#     Return a list of the zone ID's available.
    
#     """
    
#     return ZoneList(zone_ids=zones)

# @router.get("/zones/geojson", response_model=ZoneGeoJson, status_code=200, tags=["Master Data"])
# async def get_zone_boundaries(zone_geo: Dict = Depends(get_zones), zone_list: Dict = Depends(get_zone_list)):
#     """
#     Return a GEOJson payload for all zones.
    
#     """

#     filterer_geo = [ z for z in zone_geo['features'] if z['properties']['zone'] in zone_list ]

#     payload = ZoneGeoJson(features = filterer_geo)
    
    
#     return payload

# @router.post("/callouts", response_model=List[Callout], status_code=200, tags=["Master Data"])
# async def get_callouts(callouts: List = Depends(get_callouts)):
#     """
#     Return a list of callouts for a given modality.
    
#     """
#     return callouts

# @router.get("/availability", response_model=List, status_code=200, tags=["Model"])
# async def get_available_technician_ids(availabe_techs: List=Depends(get_availability)):
#     """
#     Return a list of ID's of technicians available in the rota on the specified date in the specified zone.

#     Date should be in the format: YYYY-MM-DD

#     """
#     return availabe_techs

# @router.get("/get_location_detail", response_model=LocationDetail, response_model_exclude_unset=True, status_code=200, tags=["Model"])
# async def get_details_for_lookup_text(lookup_string: str):
#     """
#     Return an object containing details for a location for a post code / post sector.

#     The lookup code should be at least 4 characters in length

#     """
    

#     try:
#         converter = PostCodeToLatLong()
#         res = converter.post_code_to_details(lookup_string.strip())
#         payload = LocationDetail(**res.dropna().to_dict())
        
#         return payload
    
#     except ValidationError:
#         raise HTTPException(status_code=404, detail="No result found")
    
#     except AssertionError:
#         raise HTTPException(status_code=400, detail="Search text should have at least 4 characters")

# @router.post("/area_of_cover", response_model=DTZGeoJson, response_model_exclude_unset=True, status_code=200, tags=["Model"])
# def generate_areas_of_cover(body: AreaOfCoverRequest):
#     """
#     Return all areas of cover for the specified zone at the specified filter settings.

#     """
#     print("Received request")
#     print(body)
#     geojson = get_geojson_list(body)

#     return geojson
