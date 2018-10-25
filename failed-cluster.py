import csv
import numpy as np
import pandas as pd
import geopandas as gpd
from geomath.hulls import ConcaveHull
from shapely.geometry import asPolygon
import mplleaflet


def load_data(cluster):
    # data_frame = pd.read_csv("data/out/failed/failed_hull_{0}.csv".format(cluster))
    data_frame = pd.read_csv("data/Marques-de-Pombal.csv")
    return data_frame


def run(cluster):
    points_df = load_data(cluster)

    # Get the underlying numpy array
    points = points_df[['Longitude', 'Latitude']].values

    # Create the concave hull object
    concave_hull = ConcaveHull(points)

    # Calculate the concave hull array
    hull_array = concave_hull.calculate(k=11)

    hull = asPolygon(hull_array)
    # buffered_hull = concave_hull.buffer_in_meters(hull, buffer)

    polygons = [hull]
    polygon_df = pd.DataFrame.from_dict(data={'polygon': polygons})
    polygon_df.to_csv("data/out/Marques-de-Pombal.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)

    # polygon_gdf = gpd.GeoDataFrame(polygon_df, geometry='polygon')
    # polygon_gdf.crs = {'init': 'epsg:4326'}
    #
    # ax = polygon_gdf.geometry.plot(linewidth=2.0, color='red', edgecolor='red', alpha=0.5)
    # mplleaflet.show(fig=ax.figure, tiles='cartodb_positron')


if __name__ == "__main__":
    run(258)
