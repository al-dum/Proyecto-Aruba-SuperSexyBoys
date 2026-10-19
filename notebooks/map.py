import json
import folium

with open('/home/igomez/WSPython/Proyecto-Aruba-SuperSexyBoys/data/raw/aruba_stations.json', 'r') as f:
    stations = json.load(f)

m = folium.Map(location=[12.52, -69.98], zoom_start=11)

folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri',
    name='Esri Satellite',
    overlay=False,
    control=True
).add_to(m)

for station in stations:
    folium.Marker(
        location=[station['latitude'], station['longitude']],
        popup=f"{station['name']} ({station['district']})",
        tooltip=station['code']
    ).add_to(m)

m.save('aruba_satellite_map.html')