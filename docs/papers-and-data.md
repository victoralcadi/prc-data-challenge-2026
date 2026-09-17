# Prior work and candidate external datasets

Prize eligibility requires every external dataset to be openly accessible, openly
licensed, and documented. This file is that documentation. Anything actually used must
also be listed in the README with its licence.

Content below is summarised and paraphrased from the linked sources.

## Papers

Ordered by how close they are to this challenge.

### Directly on taxi time at European airports

- **[Generalisable GBDT models for taxi-out and taxi-in](https://www.sesarju.eu/sites/default/files/documents/sid/2025/papers/SIDs_2025_paper_119-final%20v2.pdf)**
  (SESAR Innovation Days 2025, paper 119). Gradient boosted trees over 4.1M flights,
  2022–2024, six European airports, mostly A-CDM. Inputs are airport operational records
  plus METAR observations plus BADA aircraft characteristics.
  The finding that matters most to us: against the operational baseline of Standard Taxi
  Times, error reductions were large at regional airports but only a few percent at major
  ones. Our ten airports are all major hubs, which is consistent with the leaderboard
  being packed into a ~30s band.
- **[Variable taxi time prediction for A-CDM](https://www.sesarju.eu/sites/default/files/documents/sid/2021/papers/SIDs_2021_paper_78.pdf)**
  (SESAR Innovation Days 2021, paper 78). Framing of taxi time prediction as an A-CDM
  planning input.
- **[Taxi-Out Time Prediction at Charlotte using Machine Learning](https://ntrs.nasa.gov/api/citations/20180002162/downloads/20180002162.pdf)**
  (NASA NTRS). A standard reference for the feature families: demand counts, queue
  position, runway configuration, stand location.

### Congestion, queueing and flow control

- **[Airport Taxi Time Prediction and Alerting](https://arxiv.org/abs/2111.09139)** (arXiv
  2111.09139). Predicts whether *average* taxi-out at an airport will breach a threshold in
  the next hour. Useful for airport-hour aggregate features rather than per-flight ones.
- **[Additional taxi-out time prediction fusing flow control information](https://www.mdpi.com/2076-3417/14/21/9968)**
  (Applied Sciences). Models *additional* (excess over unimpeded) taxi-out and folds in
  flow restrictions. The unimpeded/excess decomposition is worth borrowing: predict a
  per-stand-runway unimpeded baseline, then predict the excess.
- **[Stochastic departure metering with ML taxi-time predictors](https://www.mdpi.com/2226-4310/13/8/684)**
  (Aerospace). Tokyo Haneda. Notable for treating taxi time as a distribution rather than a
  point estimate, and for right-tail risk.

### Surface geometry and microscopic features

- **[A road-level transport network model with microscopic operational features](https://www.mdpi.com/2226-4310/12/8/721)**
  (Aerospace). Builds a taxiway-graph representation. Reports that microscopic features
  moved 1-minute accuracy from ~49% to ~54%, i.e. real but not transformative gains.
- **[A CNN-GRU hybrid for departure taxiing time](https://www.mdpi.com/2226-4310/11/4/261)**
  (Aerospace) and **[a multi-module fusion model](https://www.mdpi.com/2226-4310/13/4/314)**
  (Aerospace). Deep spatio-temporal approaches. Included for completeness; on tabular
  operational data with ~2M rows, GBDTs are the stronger starting point.
- **[LSTM-XGBoost with sparrow search optimisation](https://link.springer.com/chapter/10.1007/978-3-032-08290-9_3)**
  (Springer). Hybrid sequence + boosting.

### Previous editions of this challenge

Read for method and pitfalls. **Do not copy code**: eligibility requires an original
solution, and reuse needs the authors' permission plus substantial modification.

- 2024 data paper, take-off weight: <https://journals.open.tudelft.nl/joas/article/view/8252>
- 2025 data paper, fuel burn: <https://journals.open.tudelft.nl/joas/article/view/8750>
- Winning-model write-ups: <https://journals.open.tudelft.nl/joas/article/view/7963> and
  <https://journals.open.tudelft.nl/joas/article/view/8764>
- Teams' repositories: <https://github.com/prc-data-challenge-2024> and
  <https://github.com/PRC-Data-Challenge-2025>
- A participant's comparison of the 2024 winning models:
  <https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/>

## Candidate datasets

### Weather, highest expected value

The ranking months are **January and July**. January at FRA, MUC, ZRH, CDG and AMS means
de-icing, which is applied after push-back and lands squarely inside taxi-out. Freezing
conditions and low visibility are the strongest external signals available to us.

| Source | Licence | Why |
| --- | --- | --- |
| [Iowa State IEM ASOS/METAR archive](https://mesonet.agron.iastate.edu/request/download.phtml) ([docs](https://mesonet.agron.iastate.edu/info/datasets/metar.html)) | Open, free, no key | Actual observations per ICAO station: temperature, dewpoint, wind, visibility, present weather, precipitation. Same class of input as the SESAR GBDT paper. First choice. |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | [CC BY 4.0](https://open-meteo.com/en/license) | Hourly reanalysis by lat/lon, no key, trivial to call. Good fallback and easy gap-filling. |
| [ARCO-ERA5 on Google Cloud](https://cloud.google.com/storage/docs/public-datasets/era5) | Open | What the organisers themselves used to augment the 2024 trajectories. Heavier to work with. |
| [NOAA Integrated Surface Database](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Public domain | Alternative station archive. |

Features worth deriving: freezing flag (temperature near or below 0 with precipitation),
snow present, visibility below LVP thresholds, crosswind component against the departure
runway, gusts, thunderstorm present.

### Airport surface geometry

Taxi-out is distance plus queueing. `STAND_mvt` and `RUNWAY_mvt` already encode distance
implicitly, and the group statistics in `features.py` capture the common pairs, so the
gain here is mainly for rare pairs and for generalisation.

| Source | Licence | Why |
| --- | --- | --- |
| [OpenStreetMap aeroways](https://wiki.openstreetmap.org/wiki/Aeroways), via [Overpass](https://overpass-api.de/) or Geofabrik extracts | ODbL, attribution required | Taxiway graph, apron and stand positions, runway ends. Enables true stand-to-runway distance. |
| [OurAirports data](https://ourairports.com/data/) | Public domain | Runway ends, lengths, headings, elevations. Light and easy. |

### Aircraft characteristics

| Source | Licence | Why |
| --- | --- | --- |
| [OpenAP](https://github.com/TUDelft-CNS-ATM/openap) ([paper](https://www.mdpi.com/2226-4310/7/8/104)) | Open source | Wingspan, MTOW, engines by type. The open alternative to BADA, which needs a licence and so is awkward for a GPLv3 submission. |
| [ICAO Doc 8643](https://www.icao.int/publications/doc8643/pages/search.aspx) | Reference | Type designator to wake category and engine count. |

### Network and ATFM context

| Source | Notes |
| --- | --- |
| [EUROCONTROL open data](https://www.eurocontrol.int/our-data) and [network performance](https://www.eurocontrol.int/network-performance) | Airport-level reference taxi times and delay statistics; monthly granularity limits per-flight use. |
| [EUROCONTROL R&D data archive](https://www.eurocontrol.int/dashboard/rnd-data-archive) | Released on a two-year delay, so 2025–2026 is not available for this challenge. |
| OpenSky historical ADS-B | Surface trajectories could yield observed queue positions, at high processing cost. |

## Suggested order of work

1. METAR from IEM for the ten airports, 2025 plus Jan and Jul 2026. Cheap, and targets the
   January de-icing effect directly.
2. Runway geometry from OurAirports for crosswind, then OSM stand coordinates if the
   stand-runway group statistics prove weak on rare pairs.
3. Aircraft characteristics from OpenAP, mostly to help rare aircraft types generalise.
