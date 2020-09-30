import gpxpy
import pandas as pd
import argparse
import os
# From https://github.com/nidhaloff/gpx-converter/blob/master/gpx_converter/base.py

def csv_to_gpx(input_file, lat_colname='GPS_LAT', long_colname='GPS_LONG', output_file='output.gpx'):
    """
    convert pandas dataframe to gpx
        input_file: input file
        lat_colname: name of the latitudes column
        long_colname: name of the longitudes column
        output_file: path of the output file
    """

    df = pd.read_csv(input_file).set_index("TIME")

    output_extension = os.path.splitext(output_file)[1]
    if output_extension != ".gpx":
        raise TypeError(f"Output file must be a gpx file")

    gpx = gpxpy.gpx.GPX()

    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)


    # Create points:
    for idx in df.index:
        print (lat_colname)
        print (long_colname)
        print (df.loc[idx, 'GPS_LAT'])
        print (df.loc[idx, 'GPS_LONG'])
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(df.loc[idx, lat_colname],
                                                          df.loc[idx, long_colname]))

    with open(output_file, 'w') as f:
        f.write(gpx.to_xml())
    return gpx.to_xml()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", help="Final name of time index")
    parser.add_argument("--output", "-o", help="Output file name")
    parser.add_argument("--lat_name", "-la", default = "GPS_LAT", help="Column name for latitude")
    parser.add_argument("--long_name", "-lo", default = "GPS_LONG", help="Column name for longitude")
    
    args = parser.parse_args()
    csv_to_gpx(args.input, args.lat_name, args.long_name, args.output)    