import { LayersControl, MapContainer, ScaleControl, TileLayer } from 'react-leaflet'

// Default view: the synthetic demo road (Chennai).
export const HOME = [12.9055, 80.2275]

export default function BaseMap({ center = HOME, zoom = 18, children, className = 'map' }) {
  return (
    <MapContainer center={center} zoom={zoom} maxZoom={21} className={className}>
      <LayersControl position="topright">
        <LayersControl.BaseLayer checked name="Streets">
          <TileLayer
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            maxNativeZoom={19}
            maxZoom={21}
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          />
        </LayersControl.BaseLayer>
        <LayersControl.BaseLayer name="Satellite">
          <TileLayer
            url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            maxNativeZoom={19}
            maxZoom={21}
            attribution="Imagery &copy; Esri, Maxar, Earthstar Geographics, and the GIS User Community"
          />
        </LayersControl.BaseLayer>
      </LayersControl>
      <ScaleControl position="bottomleft" imperial={false} />
      {children}
    </MapContainer>
  )
}
