# Prior work and candidate external datasets

Prize eligibility requires every external dataset to be openly accessible, openly
licensed, and documented. This file is that documentation. Anything actually used must
also be listed in the README with its licence.

Content below is summarised and paraphrased from the linked sources.

## 1. How to think about taxi-out

The literature converges on one decomposition, and the organisers use it too:

    taxi-out = unimpeded time (geometry) + additional time (queueing, congestion, weather)

The MIT work formalised this. Simaiakis and Balakrishnan model taxi-out as unimpeded
time plus departure-queue time plus ramp/taxiway interaction delay, and show the
departure runway queue is what dominates during congestion
([queuing model of the departure process](https://dspace.mit.edu/handle/1721.1/105359),
[thesis](https://dspace.mit.edu/handle/1721.1/54783),
[ACC 2014](http://www.mit.edu/~hamsa/pubs/SimaiakisBalakrishnanACC2014.pdf)). Badrinath
and Balakrishnan refine it to two queues in tandem, ramp then runway
([ACC 2017](http://www.mit.edu/~hamsa/pubs/BadrinathBalakrishnanACC2017.pdf)). A 2026
paper in Scientific Reports reports the same split empirically: unimpeded time is driven by
airport configuration, dynamic time by surface traffic flow
([Sci. Rep.](https://www.nature.com/articles/s41598-026-40898-5.pdf)).

EUROCONTROL's PRU, which runs this challenge, measures airports with exactly this
indicator. Their additional taxi-out time methodology computes a reference
(unimpeded) time per stand-group / runway pair from the 20th percentile of observed taxi
times, and reports the excess
([methodology PDF](https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf),
[definition](https://ansperformance.eu/definition/additional-taxi-out-time/)).

What this means for us: `STAND_mvt` x `RUNWAY_mvt` group statistics already approximate
the unimpeded component. The modelling effort belongs on the additional component, which
is a function of what else is happening on the surface around takeoff time, and of weather.

Scale of the target we are chasing: for 2025, average additional taxi-out was about 3.9
min/dep at Frankfurt and 3.8 at Madrid, on top of a reference of roughly 10-14 min
([EDDF dashboard](https://ansperformance.eu/dashboard/stakeholder/airport/db/eddf.html),
[LEMD dashboard](https://ansperformance.eu/dashboard/stakeholder/airport/db/lemd.html)).
The variance of the additional part is what separates 690s (our constant) from 242s
(the leaders).

## 2. Papers

### Closest to this challenge

- **[Generalisable GBDT models for taxi-out and taxi-in](https://www.sesarju.eu/sites/default/files/documents/sid/2025/papers/SIDs_2025_paper_119-final%20v2.pdf)**
  (SESAR Innovation Days 2025, paper 119). Gradient boosted trees, 4.1M flights,
  2022-2024, six European airports, mostly A-CDM. Inputs: airport operational records
  plus METAR plus BADA aircraft characteristics. Against Standard Taxi Times, error fell
  30-44% at regional airports but only 1-6% at major ones. All ten of ours are major.
  That is consistent with the leaderboard sitting in a ~30s band.
- **[Additional taxi-out time prediction fusing flow control](https://www.mdpi.com/2076-3417/14/21/9968)**
  (Applied Sciences, Shanghai Pudong). Builds stand-group x runway origin-destination
  pairs, regresses an unimpeded time per pair on arrival and departure flows, then
  predicts the residual. The cleanest published recipe for the decomposition above.
- **[Taxi-Out Time Prediction at Charlotte](https://ntrs.nasa.gov/api/citations/20180002162/downloads/20180002162.pdf)**
  (NASA). The reference feature families: demand counts, queue position, runway
  configuration, stand location.
- **[Variable taxi time prediction for A-CDM](https://www.sesarju.eu/sites/default/files/documents/sid/2021/papers/SIDs_2021_paper_78.pdf)**
  (SESAR ID 2021, paper 78; [slides](https://sesar.eu/sites/default/files/documents/sid/2021/SIDs_2021_presentation_78.pdf)).
  Splits taxi-out into apron, taxiway and runway-zone segments.

### Queueing and congestion

- [Airport Taxi Time Prediction and Alerting](https://arxiv.org/abs/2111.09139) (arXiv).
  Airport-hour aggregate: will mean taxi-out breach a threshold in the next hour.
- [Stochastic departure metering with ML taxi-time predictors](https://www.mdpi.com/2226-4310/13/8/684)
  (Aerospace, Haneda). Treats taxi time as a distribution; explicit about right-tail risk.
- [Evaluating the impact of uncertainty on surface operations](http://www.mit.edu/~hamsa/pubs/Badrinath_etal_Aviation2018.pdf)
  and [integrated surface-airspace model](https://www.mit.edu/~hamsa/pubs/BadrinathLiBalakrishnan_JGCD2019.pdf)
  (MIT). Demand exceeding capacity is the mechanism; runway configuration is a first-order
  input.
- [Atkin et al., runway sequencing at Heathrow](https://people.cs.nott.ac.uk/pszja/papers/atkin_TransportationScience2007.pdf)
  (Transportation Science 2007). Wake-turbulence separations between consecutive
  departures drive runway throughput. Motivates the `rwy_gap` features and a
  wake-category-mix feature for the preceding departures.

### Geometry and deep models

- [Road-level network model with microscopic features](https://www.mdpi.com/2226-4310/12/8/721)
  (Aerospace). Taxiway graph; 1-minute accuracy from ~49% to ~54%. Real, not
  transformative.
- [CNN-GRU](https://www.mdpi.com/2226-4310/11/4/261),
  [multi-module fusion](https://www.mdpi.com/2226-4310/13/4/314),
  [LSTM-XGBoost](https://link.springer.com/chapter/10.1007/978-3-032-08290-9_3).
  Deep spatio-temporal approaches. On ~2M rows of tabular operational data, GBDTs remain
  the stronger starting point.

### Previous editions

Read for method and pitfalls. **Do not copy code**: eligibility requires an original
solution, and reuse needs the authors' permission plus substantial modification.

- Data papers: [2024, take-off weight](https://journals.open.tudelft.nl/joas/article/view/8252),
  [2025, fuel burn](https://journals.open.tudelft.nl/joas/article/view/8750)
- Winning-model papers: [2024](https://journals.open.tudelft.nl/joas/article/view/7963),
  [2025](https://journals.open.tudelft.nl/joas/article/view/8764)
- Team repos: [2024](https://github.com/prc-data-challenge-2024),
  [2025](https://github.com/PRC-Data-Challenge-2025)
- [A participant's comparison of the 2024 winners](https://conormclaughlin.net/2024/12/reviewing-the-winning-ml-models-from-the-opensky-prc-data-challenge-what-did-they-do-differently/)

## 3. Operational mechanisms worth encoding

These are things the data cannot tell you but the airports publish.

**De-icing sits inside taxi-out.** At Frankfurt, remote de-icing pads are used and the
de-icing time is planned into the A-CDM sequence
([Frankfurt de-icing plan 2025-26](https://cdm.frankfurt-airport.com/content/dam/fraport-company-cdm/documents/binary/documents/deicing-2025-2026/EN-DIP%202025-2026.pdf),
[A-CDM webinar](https://cdm.frankfurt-airport.com/content/dam/fraport-company-cdm/documents/binary/documents/Pr%C3%A4sentation_A-CDM%20Webinar_2020.pdf)).
Munich publishes seasonal counts: 6,937 aircraft de-iced in 2024-25 and **8,628 in
2025-26**, well above their own plan of 6,900
([MUC report 24-25](https://redirect-srv.munich-airport.de/_b/0000000000000033355465bb68590aa7/De-icing-annual-report-24-25.pdf),
[MUC report 25-26](https://www.munich-airport.de/_b/0000000000000025084544bb66699085/annual-report-deicing-23-24.pdf),
[MUC plan 2025-26](https://www.munich-airport.com/_b/0000000000000035898332bb68e77735/deicing-plan-2025-2026.pdf)).
So the January 2026 ranking month was a heavier de-icing winter than the January 2025 we
train on. A freezing-conditions feature is not optional.

**Runway configuration changes the geometry.** The PRU dashboards publish each airport's
2025 configurations and usage shares. Frankfurt: 07L/07R arrivals with 07C/18 departures
47% of the time, versus 25-direction configurations 48%. Madrid: 32L/32R arrivals with
36L/36R departures 77%, the 14/18 configuration 18%. Configuration is implied by
`RUNWAY_mvt` for the flight itself, but the *airport-wide* configuration at that moment
can be inferred from the runways of neighbouring movements, and it determines which
stands are far from which runways.

**Schiphol's Polderbaan (18R/36L) is 5 km from the terminal**, giving taxi times of 15-20
minutes against ~10 for other runways
([airporthistory](https://www.airporthistory.org/blue-concourse/amsterdam-schiphols-runway-in-the-middle-of-nowhere),
[Simple Flying](https://simpleflying.com/why-is-amsterdam-schiphols-polderbaan-runway-so-far-from-the-terminal/)).
Runway choice at EHAM follows a preference system driven by wind, visibility and noise
rules ([LVNL](https://www.lvnl.nl/omgeving/actueel-baangebruik-schiphol),
[Schiphol](https://www.schiphol.nl/nl/schiphol-als-buur/geluid-en-baancombinaties/)).
Extreme case of why stand x runway matters more than either alone.

**Heathrow alternates runways at 15:00** local for noise, switching which one departs
([Heathrow](https://www.heathrow.com/company/local-community/noise/operations/runway-alternation)).
A clean, predictable discontinuity in the EGLL stand-to-runway distance by time of day.

## 4. Candidate datasets

### 4a. Weather, highest expected value

The ranking months are January and July. January at FRA, MUC, ZRH, CDG and AMS means
de-icing and low visibility; July means convective weather and thunderstorm ground stops.

| Source | Licence | Notes |
| --- | --- | --- |
| [Iowa State IEM ASOS/METAR archive](https://mesonet.agron.iastate.edu/request/download.phtml) ([docs](https://mesonet.agron.iastate.edu/info/datasets/metar.html)) | Open, free, no key | Actual per-station observations for all ten ICAO codes: temperature, dewpoint, wind, gusts, visibility, present weather, precipitation. Same input class as the SESAR GBDT paper. First choice. |
| [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) | [CC BY 4.0](https://open-meteo.com/en/license) | Hourly reanalysis by lat/lon, no key. Fallback and gap-filling. |
| [ARCO-ERA5 on Google Cloud](https://cloud.google.com/storage/docs/public-datasets/era5) | Open | What the organisers used to augment 2024 trajectories. Heavier. |
| [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Public domain | Alternative station archive. |

Derived features: freezing flag (temperature near or below 0 with any precipitation),
snow present, visibility below low-visibility-procedure thresholds, crosswind and tailwind
components against `RUNWAY_mvt` heading (needs runway headings, below), gust, thunderstorm.

### 4b. EUROCONTROL / PRU open data, covers the ranking months

The PRU's [data page](https://ansperformance.eu/data/) publishes Excel downloads that run
into 2026. Airport-level, not per-flight, but they cover January and July 2026 and so
provide *contemporaneous* context for the ranking months that the training data lacks.

| Dataset | Granularity | Coverage | Download |
| --- | --- | --- | --- |
| Taxi-out additional time | monthly, per airport | Jan 2018 - Jun 2026 | [xlsx](https://www.eurocontrol.int/performance/data/download/xls/Taxi-Out_Additional_Time.xlsx), [schema](https://ansperformance.eu/reference/dataset/taxi-out-additional-time/) |
| Airport traffic | daily, per airport | Jan 2016 - Jul 2026 | [xlsx](https://www.eurocontrol.int/performance/data/download/xls/Airport_Traffic.xlsx) |
| ATC pre-departure delay | daily, per airport | Jan 2016 - Jun 2026 | [xlsx](https://www.eurocontrol.int/performance/data/download/xls/ATC_Pre-Departure_Delay.xlsx) |
| All-causes pre-departure delay | daily, per airport | Jan 2020 - Jun 2026 | [xlsx](https://www.eurocontrol.int/performance/data/download/xls/All_Pre-Departure_Delay.xlsx) |
| Taxi-time planning values (CODA) | per IATA season | S14 - S22 | [xlsx](https://www.eurocontrol.int/performance/data/download/xls/Taxi_times_Planning_Data_S14_S22.xlsx) |

Uses: the monthly additional-taxi-out series gives a per-airport-month prior for the
excess component; daily pre-departure delay is a proxy for how disrupted a given day was.
Note the monthly series stops at June 2026, so July 2026 is not covered by it.

Licence: EUROCONTROL data is published for reuse under its
[disclaimer](https://ansperformance.eu/about/disclaimer/); confirm attribution wording
before submission.

### 4c. OPDI, joint OSN/PRC flight events. Handle with care

The [Open Performance Data Initiative](https://www.opdi.aero/) publishes flight lists,
events and measurements derived from OpenSky ADS-B, covering
**January 2022 - July 2026** ([data](https://www.opdi.aero/data),
[concepts](https://www.opdi.aero/concepts)). The event types include `off-block` with the
parking position as `info`, `end of push back`, `enter runway for take-off` with the
runway ID, and `lift-off`.

That is the blanked column, reconstructed from ADS-B, for the ranking months, published by
the organisers' own sister initiative. Two readings are possible:

1. It is open data, the rules allow open data, and matching OPDI off-block events to
   ranking departures is legitimate feature engineering.
2. It defeats the purpose of the challenge, and the ranking page warns that attempts to
   "learn from or exploit the ranking process" are considered unfair.

Do not use it for the blanked quantity without asking on the `#prc-data-competition`
Discord channel first. There *are* uncontroversial uses: OPDI's `enter runway` and
`lift-off` events for *other* aircraft give observed runway queue lengths, and its runway
IDs give the airport-wide configuration at any moment. Also useful for validation on 2025:
compare OPDI-derived taxi-out against `TAXITIME_SEC_mvt` to understand how the reporting
airports measure it.

OPDI also republishes OurAirports [airport](https://www.opdi.aero/data) and runway tables
as CSV, which covers 4d below.

### 4d. Airport geometry

| Source | Licence | Why |
| --- | --- | --- |
| [OurAirports](https://ourairports.com/data/) | Public domain | Runway ends, headings (`le_heading_degT`), lengths, threshold coordinates. Needed for crosswind. Also via OPDI. |
| [OpenStreetMap aeroways](https://wiki.openstreetmap.org/wiki/Aeroways) via [Overpass](https://overpass-api.de/) or Geofabrik | ODbL, attribution required | Taxiway graph, apron and stand positions. Enables true stand-to-runway distance for rare stand x runway pairs the group statistics cover poorly. |

### 4e. Aircraft characteristics

| Source | Licence | Why |
| --- | --- | --- |
| [OpenAP](https://github.com/TUDelft-CNS-ATM/openap) ([paper](https://www.mdpi.com/2226-4310/7/8/104)) | Open source | Wingspan, MTOW, engine count by type. The open alternative to BADA, which needs a licence and is awkward alongside a GPLv3 submission. |
| [ICAO Doc 8643](https://www.icao.int/publications/doc8643/pages/search.aspx) | Reference | Type designator to wake category and engine count. |

Mostly helps rare aircraft types generalise; `WK_TBL_CAT_flt` already carries the wake
category for matched flights.

### 4f. Not usable

- [EUROCONTROL R&D data archive](https://www.eurocontrol.int/dashboard/rnd-data-archive):
  released on a two-year delay, so nothing for 2025-2026.
- BADA: licence terms, see above.

## 5. Suggested order of work

1. Baseline first. `prc2026 train` with the internal features only, so every external
   addition can be measured against a number.
2. METAR from IEM for the ten airports, 2025 plus Jan and Jul 2026. Cheap, and targets
   the January de-icing effect directly. Add runway headings from OurAirports at the same
   time for crosswind.
3. PRU monthly additional-taxi-out and daily pre-departure delay as airport-day context,
   especially for the ranking months.
4. Infer airport-wide runway configuration from neighbouring movements' `RUNWAY_mvt`.
5. Ask on Discord about OPDI before touching its off-block events. Use its runway-entry
   events for queue length regardless.
6. OSM stand coordinates only if stand x runway group statistics prove weak on rare pairs.
