import geopandas
from models_train.get_models import DeepModels

def test_if_opening_gpkg_return_geodataframe():
    model = DeepModels()

    assert type(model.run_geopackage('amostra','gpkg')) == geopandas.geodataframe.GeoDataFrame

