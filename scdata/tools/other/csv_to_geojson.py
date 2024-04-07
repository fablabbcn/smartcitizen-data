import pandas
import os
import numpy
from geojson import Point, Feature, FeatureCollection, dumps
import argparse

# From https://github.com/miquel-vv/csv-to-geojson/blob/master/

def csv_to_geojson(input_file, lat_colname='GPS_LAT', long_colname='GPS_LONG', output_file='output.geojson'):
    fold = os.path.dirname(input_file)
    name,_ = os.path.basename(input_file).split('.')
    
    output_extension = os.path.splitext(output_file)[1]
    if output_extension != ".geojson":
        raise TypeError(f"Output file must be a geojson file")
    
    df = pandas.read_csv(input_file).fillna('')
    lat = df[lat_colname]
    lng = df[long_colname]
    df = df.drop(columns=[lat_colname, long_colname])
    
    feat_list = []
    failed = []
    for i in range(0, len(df.index)):
        props = remove_np_from_dict(dict(df.loc[i]))
        try:
            f = Feature(geometry=Point((float(lng[i]), float(lat[i]))),
                       properties = props)
            feat_list.append(f)
        except ValueError:
            failed.append(props)
        
    collection = FeatureCollection(feat_list)
    with open(output_file, 'w') as f:
        f.write(dumps(collection))
    
    return output_file

def remove_np_from_dict(d):
    '''numpy int64 objects are not serializable so need to convert values first.'''
    new={}
    for key, value in d.items():
        if isinstance(key, numpy.int64):
            key = int(key)
        if isinstance(value, numpy.int64):
            value = int(value)
        new[key] = value
    return new
    
def convert_numpy(val):
    if isinstance(val, numpy.int64): return int(val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Final name of time index")
    parser.add_argument("--output", "-o", help="Output file name")
    parser.add_argument("--lat_name", "-la", default = "GPS_LAT", help="Column name for latitude")
    parser.add_argument("--long_name", "-lo", default = "GPS_LONG", help="Column name for longitude")
    
    args = parser.parse_args()
    csv_to_geojson(args.input, args.lat_name, args.long_name, args.output) 