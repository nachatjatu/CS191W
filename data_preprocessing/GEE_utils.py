
import ee

ee.Authenticate()
ee.Initialize(project='ee-thailand-pm')


def get_pm25(pm25_ic, date):
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
    fires = fire_ic.filterDate(date, date.advance(1, 'day')).first()

    fire_scale = fires.projection().nominalScale()

    # correct for area-weighting in reduceResolution
    correc_fac = weather_scale.pow(2).divide(fire_scale.pow(2))
    
    # filter only high-confidence fire pixels
    mask = fires.select('FireMask').gte(8)
    frp = fires.updateMask(mask).select('MaxFRP').unmask(0)

    frp_lr = frp.reduceResolution(
        reducer=ee.Reducer.mean(),
        maxPixels=1024
    ).multiply(correc_fac)

    return frp_lr.rename('frp')


def get_weather(weather_ic, date):
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