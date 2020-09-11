from datetime import datetime
import numpy as np
import joblib
import pandas as pd
import streamlit as st
import geopandas as gpd
from TaxiFareModel.data import get_data
import geoviews as gv

gv.Polygons(gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')), vdims=['pop_est', ('name', 'Country')]).opts(
    tools=['hover'], width=600, projection=crs.Robinson()
)

st.markdown("# ML Project")
st.markdown("**Taxifare data explorer**")

@st.cache
def read_data(n_rows=10000):
    df = get_data(n_rows=n_rows, local=False)
    return df


def format_input(pickup, dropoff, passengers=1):
    pickup_datetime = datetime.utcnow().replace(tzinfo=pytz.timezone('America/New_York'))
    formated_input = {
        "pickup_latitude": pickup["latitude"],
        "pickup_longitude": pickup["longitude"],
        "dropoff_latitude": dropoff["latitude"],
        "dropoff_longitude": dropoff["longitude"],
        "passenger_count": passengers,
        "pickup_datetime": str(pickup_datetime),
        "key": str(pickup_datetime)}
    return formated_input


def main():
    analysis = st.sidebar.selectbox("chose restitution", ["Dataviz", "Prediction"])
    if analysis == "Dataviz":
        df = pd.DataFrame(
            {'City': ['Buenos Aires', 'Brasilia', 'Santiago', 'Bogota', 'Caracas'],
             'Country': ['Argentina', 'Brazil', 'Chile', 'Colombia', 'Venezuela'],
             'Latitude': [-34.58, -15.78, -33.45, 4.60, 10.48],
             'Longitude': [-58.66, -47.91, -70.66, -74.08, -66.86]})
        gdf = geopandas.GeoDataFrame(
            df, geometry=geopandas.points_from_xy(df.Longitude, df.Latitude))
        st.write(gdf.head())
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        ax = world[world.continent == 'South America'].plot(
            color='white', edgecolor='black')
        gdf.plot(ax=ax, color='red')
        st.pyplot()

    if analysis == "prediction":
        pipeline = joblib.load('data/model.joblib')
        print("loaded model")
        st.header("TaxiFare Model predictions")

        st.write("💸 taxi fare", res[0])
        st.map(data=data)


# print(colored(proc.sf_query, "blue"))
# proc.test_execute()
if __name__ == "__main__":
    #df = read_data()
    main()
