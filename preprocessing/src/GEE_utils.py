import ee

def get_pm25(pm25_ic, date):
    """retrieves PM2.5 and Delta PM2.5 images on date

    Args:
        pm25_ic (ee.ImageCollection): ImageCollection for PM2.5 data
        date (ee.Date): date of data retrieval

    Returns:
        ee.Image, ee.Image: PM2.5 image, Delta PM2.5 image on date
    """
    pm25_today = pm25_ic.filterDate(date).first()
    pm25_tomorrow = pm25_ic.filterDate(date.advance(1, 'day')).first()

    pm25_change = pm25_tomorrow.subtract(pm25_today)

    pm25_today_lr = pm25_today.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=1024
    ).rename('pm25_today')

    pm25_change_lr = pm25_change.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=1024
    ).rename('pm25_change')

    return pm25_today_lr, pm25_change_lr


def get_fire(fire_ic, weather_scale, fire_scale, date):
    """retrieves fire detection image on date at weather_scale resolution

    Args:
        fire_ic (ee.ImageCollection): ImageCollection for fire detection data
        weather_scale (ee.Number): scale of weather data
        fire_scale (ee.Number): scale of fire data
        date (ee.Date): date of data retrieval

    Returns:
        ee.Image: fire detection image
    """
    fires = fire_ic.filterDate(date, date.advance(1, 'day')).first()

    fire_scale = fires.projection().nominalScale()

    # correct for area-weighting in reduceResolution
    correc_fac = weather_scale.pow(2).divide(fire_scale.pow(2))
    
    # filter only medium/high-confidence fire pixels
    mask = fires.select('FireMask').gte(8)
    frp = fires.updateMask(mask).select('MaxFRP').unmask(0)

    frp_lr = frp.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=1024
    ).multiply(correc_fac)

    return frp_lr.rename('frp')


def get_weather(weather_ic, date):
    """retrieves weather image on date

    Args:
        weather_ic (ee.ImageCollection): ImageCollection for weather data (ERA5-Land)
        date (ee.Date): date of data retrieval

    Returns:
        ee.Image: multi-band image with weather variables on date
    """
    weather = weather_ic.filterDate(date, date.advance(1, 'day')).first()
    
    bands = ['u_component_of_wind_10m', 'v_component_of_wind_10m', 
             'dewpoint_temperature_2m', 'temperature_2m', 'surface_pressure']
    
    bands_renamed = ['u_wind_10m', 'v_wind_10m', 
             'dew_temp_2m', 'temp_2m', 'surf_pressure', 'precip_sum']
    
    precip = weather.select('total_precipitation_sum').max(0)

    return weather.select(bands).addBands(precip).rename(bands_renamed)


NULL = ee.Image.constant(0).updateMask(0)

def create_date_list(start_date, end_date):
    """
    creates a List of sequential Dates and returns the result

    Args:
    start_date   (ee.Date):  start date to filter by
    end_date     (ee.Date):  end date to filter by

    Returns:
    ee.List[ee.Date]: the list of sequential Dates
    """
    days_seq = ee.List.sequence(0, end_date.difference(start_date, 'day'))
    return days_seq.map(lambda day: start_date.advance(day, 'day'))


def get_data_date(ics, terrain_img, weather_scale, fire_scale, date):
    """retrives data on date, returning an Image

    Args:
        ics (dict[str: ee.ImageCollection]): dictionary containing dataset ImageCollections
        terrain_img (ee.Image): DEM image (SRTM-v4)
        weather_scale (ee.Number): scale of weather data
        fire_scale (ee.Number): scale of fire data
        date (ee.Date): date of data retrieval

    Returns:
        ee.Image: complete image with multiple bands, one for each variable.
    """
    pm25_today_lr, pm25_change_lr = get_pm25(ics['pm25'], date)
    weather = get_weather(ics['weather'], date)
    frp_lr = get_fire(ics['fire'], weather_scale, fire_scale, date)

    combined = (
        ee.Image.cat(pm25_today_lr, weather, frp_lr, terrain_img, pm25_change_lr)
                .set('system:time_start', date.millis())
                .toFloat()
    )

    num_bands = combined.bandNames().length()
    date = date.format('YYYY-MM-dd')
    id = ee.String('CS191W_DSET_').cat(ee.String(date))

    combined =  (combined.set('date', date)
                .set('id_no', id)
                .set('num_bands', num_bands)
                .set('id_no', id)
                .set('system:index', id))

    return combined

def create_date_image(ics, terrain_img, weather_scale, fire_scale, date):
    """retrieves data for all variables on date, returning a multi-band Image or NULL (if missing)

    Args:
        ics (dict[str: ee.ImageCollection]): dictionary containing dataset ImageCollections
        terrain_img (ee.Image): DEM image (SRTM-v4)
        weather_scale (ee.Number): scale of weather data
        fire_scale (ee.Number): scale of fire data
        date (ee.Date): date of data retrieval

    Returns:
        ee.Image: multi-band image with complete data on date. If missing data, returns NULL image.
    """
    date = ee.Date(date)

    fires = ics['fire'].filterDate(date)
    has_fire = fires.size().gt(0)

    pm25_today = ics['pm25'].filterDate(date)
    has_pm25_today = pm25_today.size().gt(0)

    pm25_tomorrow = ics['pm25'].filterDate(date.advance(1, 'day'))
    has_pm25_tomorrow = pm25_tomorrow.size().gt(0)

    return ee.Algorithms.If(
        has_pm25_today.And(has_pm25_tomorrow).And(has_fire),
        get_data_date(ics, terrain_img, weather_scale, fire_scale, date),
        NULL
    )