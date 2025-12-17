# Wind Spatio-Temporal Dataset - Complete Information

## Dataset Overview
This is a **wind farm dataset** containing spatio-temporal measurements from a wind farm with **200 wind turbines** and **3 meteorological masts**.

---

## Key Dataset Characteristics

### 📊 Size & Structure
- **Total Records:** 8,764 hourly observations
- **Time Period:** September 1, 2010 to August 31, 2011 (1 full year)
- **Temporal Resolution:** Hourly measurements
- **Total Columns:** 606 features

### 🌍 Spatial Coverage
- **Wind Turbines:** 200 units (Turbine1 through Turbine200)
- **Meteorological Masts:** 3 measurement stations (Mast1, Mast2, Mast3)
- **Geographic Location:** 
  - Latitude range: ~40.4°N to 40.48°N
  - Longitude range: ~-88.8°W to -88.48°W
  - Location: Illinois, USA region

---

## Data Structure

### Column Organization

#### 1. Geographic Coordinates (Rows 2-3)
- **Row 2:** Latitude coordinates for all 200 turbines + 3 masts
- **Row 3:** Longitude coordinates for all 200 turbines + 3 masts

#### 2. Time Series Data (Row 5 onwards)
- **Time column:** Timestamp in format "M/D/YYYY H:MM"

#### 3. For Each Turbine (400 columns total)
- **Turbine_Speed:** Wind speed measurement (m/s)
- **Turbine_Power:** Power output (normalized, 0-1 range, with some NA values)

#### 4. For Each Mast (6 columns total)
- **Mast_Speed:** Wind speed measurement (m/s)
- **Mast_Direction:** Wind direction (degrees, 0-360)

---

## Data Features

### Wind Speed
- **Range:** Approximately 1.5 to 12+ m/s
- **Characteristics:** Many low-speed readings indicating variable wind conditions
- **Measurement Frequency:** Hourly

### Power Output
- **Range:** Normalized values between 0 and 1
- **Missing Data:** Contains NA values (indicating turbine downtime or maintenance)
- **Behavior:** Shows typical wind turbine power curve behavior (lower speeds = lower power)

### Wind Direction
- **Availability:** Only for meteorological masts
- **Range:** 0-360 degrees
- **Purpose:** Understanding wind patterns across the farm

---

## Potential Use Cases for Your Project

### 1. Spatio-Temporal Forecasting
- Predict power output for each turbine based on historical patterns
- Forecast wind speeds across the farm
- Account for spatial correlations between nearby turbines

### 2. Wake Effect Analysis
- Study how upstream turbines affect downstream turbines
- Analyze wind speed degradation patterns
- Optimize turbine layout

### 3. Anomaly Detection
- Identify turbine underperformance
- Detect maintenance needs
- Find unusual patterns in power generation

### 4. Spatial Interpolation
- Estimate wind speeds at unmeasured locations
- Create wind speed heatmaps
- Model spatial dependencies

### 5. Machine Learning Applications
- Deep learning for time series forecasting
- Graph neural networks for spatial relationships
- Attention mechanisms for important turbine identification
- Recurrent neural networks (LSTM, GRU) for temporal patterns
- Convolutional neural networks for spatial features

---

## Data Quality Notes

### Missing Values
- **Presence:** "NA" values in power output columns
- **Cause:** Turbine downtime, maintenance, or sensor failures
- **Impact:** Need for imputation or careful handling in models

### Data Completeness
- **Full year:** Hourly data from Sept 2010 to Aug 2011
- **Expected hours:** 8,760 hours in a year
- **Actual records:** 8,764 (slight overlap, possibly due to daylight saving time)

### Data Consistency
- **Temporal:** Regular hourly intervals maintained throughout
- **Spatial:** Fixed turbine positions with GPS coordinates

---

## Statistical Summary

| Metric | Value |
|--------|-------|
| **Temporal Coverage** | 100% complete hourly data for one year |
| **Spatial Density** | 200 turbines with precise GPS coordinates |
| **Variables per Turbine** | 2 (speed + power) |
| **Variables per Mast** | 2 (speed + direction) |
| **Total Turbine Measurements** | 1,753,200 data points |
| **Total Mast Measurements** | 52,584 data points |
| **Total Dataset Size** | 1,805,784+ individual measurements |

---

## File Details

- **Filename:** `Wind Spatio-Temporal Dataset2.csv`
- **Format:** CSV (Comma-Separated Values)
- **Size:** Approximately 8,765 rows × 606 columns
- **Encoding:** Standard text encoding

---

## Recommended Data Processing Steps

### 1. Data Loading
```python
# Load with proper handling of coordinates
# Skip first 3 rows for time series analysis
# Or extract coordinates separately
```

### 2. Data Cleaning
- Handle NA values in power output
- Verify timestamp continuity
- Check for outliers in wind speed and power

### 3. Feature Engineering
- Calculate temporal features (hour, day, month, season)
- Compute spatial distances between turbines
- Create lag features for time series
- Extract wind direction sectors

### 4. Exploratory Analysis
- Visualize turbine locations on a map
- Plot time series for sample turbines
- Analyze power curves (speed vs power)
- Study spatial correlations

---

## Research Opportunities

This dataset is excellent for:

✅ **Wind energy forecasting** - Short-term and medium-term predictions  
✅ **Spatial modeling** - Understanding geographic dependencies  
✅ **Complex turbine interactions** - Wake effects and interference patterns  
✅ **Renewable energy optimization** - Maximizing farm efficiency  
✅ **Climate pattern analysis** - Seasonal and diurnal variations  
✅ **Predictive maintenance** - Identifying failure patterns  
✅ **Multi-variate time series** - 200+ parallel time series  
✅ **Spatio-temporal deep learning** - State-of-the-art model development

---

## Citation and Usage

**Dataset Period:** September 1, 2010 - August 31, 2011  
**Location:** Wind farm in Illinois region, USA  
**Resolution:** Hourly measurements  
**Purpose:** Research and analysis of wind energy generation patterns

---

*Document created: December 3, 2025*  
*Dataset: Wind Spatio-Temporal Dataset2.csv*
