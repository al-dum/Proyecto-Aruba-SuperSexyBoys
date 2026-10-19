# Justificación Técnica de Umbrales de Seguridad y Alerta Meteorológica
## Gemelo Digital de Aruba: Sistema de Predicción y Despacho Energético

Este documento establece la justificación técnica, climática y operativa para los umbrales de seguridad y alerta aplicados a los atributos del archivo de lecturas meteorológicas (`aruba_lecturas.json`) en la isla de Aruba [7, 15, 169, 170]. 

El diseño de estos umbrales responde directamente a la vulnerabilidad del sistema energético insular aislado de Aruba [132], operado por **WEB Aruba N.V.** (generación y desalinización) y **N.V. ELMAR** (distribución) [134], donde variables como la velocidad del viento y la temperatura determinan la estabilidad de la red (60 Hz) [148], el balance entre fuentes renovables y respaldo térmico (motores RECIP / Wärtsilä) [48, 136], y la demanda masiva de refrigeración [15, 141].

---

## 1. Resumen Ejecutivo de Umbrales

| Métrica (`aruba_lecturas.json`) | Rango Normal | Umbral de Alerta Moderada | Umbral de Alerta Crítica | Impacto Principal |
| :--- | :--- | :--- | :--- | :--- |
| `temperature_c` (°C) | 27.3 °C – 29.5 °C | > 30.3 °C | > 33.0 °C | Disparo de demanda por A/C (~50% del total) y picos de carga en red [15, 109, 141]. |
| `humidity_pct` (%) | 65% – 75% | > 80% | > 90% | Incremento de sensación térmica y aceleración de corrosión salina en aisladores y turbinas [9, 105]. |
| `wind_speed_kmh` (km/h) | 20 km/h – 35 km/h (~27.4 km/h) | < 25 km/h | < 15 km/h ó > 65–80 km/h | Déficit eólico (Vader Piet 30 MW) / Desconexión por *cut-out* en vientos extremos [138, 168]. |
| `wind_direction_deg` (°) | 80° – 110° (Alisios del E/ENE) | < 60° ó > 120° | < 40° ó > 140° | *Wind Reversal* (reversión del viento), indicador de sistemas ciclónicos o tormentas [128]. |
| `pressure_hpa` (hPa) | 1013 – 1016 hPa | < 1008 hPa | < 1000 – 1004 hPa | Inestabilidad barométrica y aproximación de depresión/tormenta tropical [96]. |
| `precipitation_mm` (mm/h) | 0.0 mm/h | > 5.0 mm/h | > 25.0 mm/h | Caída de generación solar por sombreado / Inundación repentina en infraestructura eléctrica [96]. |
| `visibility_km` (km) | 10.0 – 12.0 km | < 8.0 km | < 4.0 km | Polvo del Sahara (*Saharan Dust*), afectación al aeropuerto Reina Beatrix y menor irradiancia [85, 98]. |
| `uv_index` | 8 – 10 (Mediodía) | ≥ 11 | ≥ 12+ | Radiación extrema, estrés térmico en campo y degradación de revestimientos fotovoltaicos [13]. |

---

## 2. Justificación Detallada por Métrica

### 2.1. `temperature_c` (Temperatura Ambiental en °C)
* **Línea de Base Promedio:** La temperatura media anual de Aruba es de **28.5 °C**, con variaciones estacionales muy estrechas entre 27.3 °C (meses frescos) y 29.5 °C (meses cálidos) [97, 104].
* **Umbral de Alerta Moderada (> 30.3 °C):** Según las proyecciones climáticas del *Royal Netherlands Meteorological Institute* (KNMI), una temperatura media diaria superior a **30.3 °C** se clasifica formalmente como un "día caluroso" (*hot day*) [109]. En Aruba, la climatización (aire acondicionado y refrigeración) representa aproximadamente el **50% del consumo eléctrico total de la isla** [15, 141]. Superar los 30.3 °C desencadena un encendido masivo de equipos A/C en los sectores comercial (38%) y residencial (35%) [140, 141].
* **Umbral de Alerta Crítica (> 33.0 °C):** Temperaturas extremas por encima de 33.0 °C generan sobrecargas térmicas en los transformadores de distribución de N.V. ELMAR y reducen la eficiencia de conversión de los paneles fotovoltaicos (Parque Solar Sunrise y autoconsumo) [134, 147]. Adicionalmente, obligan a activar capacidad térmica adicional en la planta central de WEB Aruba (motores Wärtsilä 50DF / RECIP) para mantener el margen de reserva [137, 138].

---

### 2.2. `humidity_pct` (Humedad Relativa en %)
* **Línea de Base Promedio:** El rango habitual se ubica entre **65% y 75%**, representativo de un clima tropical semiárido [97].
* **Umbral de Alerta Moderada (> 80%):** Altas concentraciones de humedad atmosférica combinadas con temperaturas elevadas incrementan la sensación térmica percibida por la población [105], prolongando la curva de demanda eléctrica nocturna.
* **Umbral de Alerta Crítica (> 90%):** Dada la ubicación costera e insular de Aruba, una humedad relativa persistente superior al 90% actúa como vehículo para la acumulación de aerosol salino (*salt spray*) [9]. Esto genera un alto riesgo de **corrosión galvánica y arcos eléctricos en la aislación de líneas de distribución de alta tensión**, requiriendo un monitoreo preventivo en subestaciones y en los aerogeneradores expuestos [9, 65].

---

### 2.3. `wind_speed_kmh` (Velocidad del Viento en km/h)
* **Línea de Base Promedio:** El viento medio en Aruba es constante, promediando **7.6 m/s (~27.4 km/h)** movido por los vientos alisios continuos [97, 117].
* **Umbral Crítico por Déficit Eólico (< 15 km/h / ~4.1 m/s):** Corresponde a niveles donde la generación del Parque Eólico Vader Piet (30 MW) cae por debajo del **15% de su capacidad nominal** (alrededor de 4,5 MW) [168]. A esta velocidad, las turbinas operan cerca de su límite inferior de arranque (*cut-in speed*), reduciendo bruscamente la inyección de energía limpia [168]. Para evitar fluctuaciones de frecuencia en la red de 60 Hz, el Gemelo Digital debe emitir una orden previa de **despacho térmico del 100%** para arrancar los motores recíprocos fijos de WEB Aruba (RECIP I–IV) [48, 168].
* **Umbral Moderado por Déficit (< 25 km/h / ~6.9 m/s):** La producción eólica se ubica entre el 15% y el 40% de su capacidad [168]. Se requiere compensación dinámica mediante almacenamiento en baterías (BESS) o rampa fotovoltaica [148, 182].
* **Umbral Crítico por Viento Extremo / Desconexión (> 65–80 km/h / > 18–22 m/s):** Velocidades de viento que se aproximan a la velocidad de corte mecánico (*cut-out*) de las palas eólicas. Las turbinas frenan automáticamente por protección, generando un evento de *ramp-down* masivo y repentino que requiere el soporte inmediato de sistemas de inercia o flywheels [22, 148].

---

### 2.4. `wind_direction_deg` (Dirección del Viento en Grados °)
* **Línea de Base Promedio:** Entre **80° y 110°** (vientos alisios predominantes que soplan de manera estable desde el Este / Estenoreste) [97, 117].
* **Umbral de Alerta Moderada (< 60° ó > 120°):** Desviación moderada en el patrón habitual de circulación costera.
* **Umbral Crítico por Reversión del Viento (*Wind Reversal*: < 40° ó > 140°):** Una alteración severa en la dirección del viento (vientos del Oeste, Suroeste o Sur) señala la presencia de una **reversión del viento** (*wind reversal*) [128]. Meteorológicamente, este fenómeno acompaña la aproximación de sistemas de baja presión, ondas tropicales o ciclones en el Caribe [96, 128]. Al perder la ventilación natural del Este, las temperaturas en zonas urbanas aumentan significativamente y se generan marejadas en las costas protegidas del oeste [96].

---

### 2.5. `pressure_hpa` (Presión Atmosférica en hPa)
* **Línea de Base Promedio:** **1013 – 1016 hPa**, correspondiente a la presión barométrica estándar al nivel del mar en la franja ecuatorial/caribeña [85, 97].
* **Umbral de Alerta Moderada (< 1008 hPa):** Pérdida de presión que indica inestabilidad atmosférica local o convergencia de humedad.
* **Umbral de Alerta Crítica (< 1000 – 1004 hPa):** Caída barométrica severa vinculada a la proximidad de una **depresión tropical, tormenta o huracán** [96]. Aunque Aruba está ubicada al sur del cinturón principal de huracanes, estos eventos provocan precipitaciones intensas, marejadas ciclónicas y riesgos directos sobre las tuberías y la infraestructura costera de la planta de agua y energía [49, 96].

---

### 2.6. `precipitation_mm` (Precipitación Acumulada en mm/h)
* **Línea de Base Promedio:** **0.0 mm/h** la mayor parte del año. Aruba posee un clima semiárido con un promedio anual de apenas 498 mm de lluvia [97, 111].
* **Umbral de Alerta por Sombreado Solar (> 5.0 mm/h):** Precipitaciones moderadas asociadas a densa cobertura nubosa. Provocan caídas abruptas en la curva de generación de los parques solares (como el Parque Solar Sunrise de 6.1 MW y techos solares) [138, 147], requiriendo un incremento rápido en la generación de respaldo.
* **Umbral de Alerta Crítica por Inundación (> 25.0 mm/h):** Lluvias torrenciales de alta intensidad. Dado el suelo árido con baja capacidad de absorción en Aruba, este volumen genera escorrentías e inundaciones repentinas en zonas bajas [96], amenazando líneas subterráneas y subestaciones de distribución de N.V. ELMAR [65, 96].

---

### 2.7. `visibility_km` (Visibilidad Horizontal en km)
* **Línea de Base Promedio:** **10.0 – 12.0 km** bajo condiciones de atmósfera limpia [85, 97].
* **Umbral de Alerta Moderada (< 8.0 km):** Reducción ocasionada por intrusión de nubes de **Polvo del Sahara** (*Saharan Dust*) o bruma marina (*calima*).
* **Umbral de Alerta Crítica (< 4.0 km):** Visibilidad severamente comprometida por polvo denso o lluvias extremas. Impacta directamente la seguridad de las operaciones en el **Aeropuerto Internacional Reina Beatrix** (donde se ubica la estación meteorológica principal) [85, 124] y atenúa de forma significativa la irradiancia solar global horizontal (GHI), reduciendo la eficiencia fotovoltaica.

---

### 2.8. `uv_index` (Índice Ultravioleta)
* **Línea de Base Promedio (Horas Pico):** **8 – 10 (Muy Alto)**, dado que Aruba recibe más de 2.500 horas de sol anuales [2, 5].
* **Umbral de Alerta Extrema (≥ 11):** Nivel extremadamente alto de radiación UV [13]. Representa riesgos inmediatos para la salud del personal operativo en campo y acelera la fotodegradación de materiales poliméricos, aislantes de red y cubiertas de paneles solares expuestos [13].

---

## 3. Conclusión e Integración en el Gemelo Digital

La calibración de estos umbrales permite al **Gemelo Digital de Aruba** pasar de una monitorización pasiva a un **sistema reactivo de despacho energético de alerta temprana** [167, 168]. Al cruzar las lecturas en tiempo real (`aruba_lecturas.json`) con estos límites operacionales, el sistema puede predecir caídas eólicas o picos de demanda y activar oportunamente los motores recíprocos de respaldo (RECIP) o los sistemas BESS antes de que se produzca un desbalance de frecuencia o una interrupción del servicio eléctrico en la isla [48, 168, 182].
