import cdsapi
c=cdsapi.Client()

for i in range(2016,2017):
  c.retrieve('reanalysis-era5-single-levels',
  {

        'product_type':'reanalysis',
        'format':'netcdf',
        'variable':[
            '100m_u_component_of_wind','100m_v_component_of_wind',
            '2m_dewpoint_temperature','2m_temperature','surface_pressure'
        ],
        'time':'00:00',
        'day':[
            '01','02','03',
            '04','05','06',
            '07','08','09',
            '10','11','12',
            '13','14','15',
            '16','17','18',
            '19','20','21',
            '22','23','24',
            '25','26','27',
            '28','29','30',
            '31'
        ],
        'area':['0','10','-40','42'],
        'month':['09'],
        'year': "%d"%i,
    },
    'era5_nearsurface_%d.nc'%i)
