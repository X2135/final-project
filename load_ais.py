import pandas as pd
import os
import numpy as np

def wrap_lat(lat):
    """Constrain latitude to [-90, 90]"""
    lat = np.asarray(lat, dtype=float)
    return np.clip(lat, -90.0, 90.0)

def wrap_lon(lon):
    """Constrain longitude to [-180, 180]"""
    lon = np.asarray(lon, dtype=float)
    return np.mod(lon + 180.0, 360.0) - 180.0

def wrap_cog(cog):
    """Constrain COG to [0, 360)"""
    cog = np.asarray(cog, dtype=float)
    return np.mod(cog, 360.0)

def clip_sog_kn(sog, low=0.0, high=45.0):
    """Clip SOG to reasonable range (knots)"""
    sog = np.asarray(sog, dtype=float)
    return np.clip(sog, low, high)

def load_ais_data(file_path):
    """
    Load AIS data and split trajectories by MMSI.
    
    Returns: 
        dict{MMSI: {'time':[], 'lat':[], 'lon':[], 'sog':[], 'cog':[], 'length_m': float|nan}}
    """

    df = pd.read_csv(file_path, low_memory=False)
    
    df.rename(columns={
        'MMSI': 'mmsi',
        'BaseDateTime': 'time',
        'LAT': 'lat',
        'LON': 'lon',
        'SOG': 'sog',
        'COG': 'cog'
    }, inplace=True)

    df['time'] = pd.to_datetime(df['time'])
    
    grouped = df.groupby('mmsi')
    
    trajectories = {}
    for mmsi, group in grouped: 
        group = group.sort_values('time')
        
        lat_clean = wrap_lat(group['lat'].tolist())
        lon_clean = wrap_lon(group['lon'].tolist())
        sog_clean = clip_sog_kn(group['sog'].tolist())
        cog_clean = wrap_cog(group['cog'].tolist())

        length_m = np.nan
        length_col = None
        if 'Length' in group.columns:
            length_col = 'Length'
        else:
            for c in group.columns:
                if str(c).strip().lower() in ('length', 'length_m', 'ship_length_m', 'ship_length'):
                    length_col = c
                    break

        if length_col is not None:
            L = pd.to_numeric(group[length_col], errors='coerce')
            L = L[(L.notna()) & (L > 0)]
            if len(L) > 0:
                try:
                    length_m = float(L.median())
                except Exception:
                    length_m = float(L.iloc[0])
        
        trajectories[mmsi] = {
            'time': group['time'].tolist(),
            'lat': lat_clean,
            'lon': lon_clean,
            'sog': sog_clean,
            'cog': cog_clean,
            'length_m': length_m,
        }
    
    return trajectories

if __name__ == "__main__":
    print("="*50)
    print("AIS Data Loading and Statistics")
    print("="*50)
    
    ais_file = "/Users/lcx/Desktop/EN/EN/ais_sample2.csv"
    if not os.path.exists(ais_file):
        print(f"❌ AIS file not found: {ais_file}")
        exit(1)
    
    print(f"📁 Loading file: {ais_file}")
    trajs = load_ais_data(ais_file)
    
    total_points = sum(len(traj['lat']) for traj in trajs.values())
    ship_count = len(trajs)
    
    print(f"\n📊 Data Statistics:")
    print(f"  Vessel count: {ship_count} ships")
    print(f"  Total data points: {total_points:,} points")
    print(f"  Average points per ship: {total_points/ship_count:.1f} points")
    
    point_counts = [len(traj['lat']) for traj in trajs.values()]
    print(f"  Point count range: {min(point_counts)} ~ {max(point_counts)} points")
    
    print(f"\n🔍 Details for first 3 ships:")
    for i, (ship_id, traj) in enumerate(list(trajs.items())[:3]):
        print(f"  Vessel {ship_id}: {len(traj['lat'])} points")
        if len(traj['lat']) > 0:
            print(f"    Position range: Latitude {min(traj['lat']):.4f}~{max(traj['lat']):.4f}")
            print(f"                    Longitude {min(traj['lon']):.4f}~{max(traj['lon']):.4f}")    
            print(f"\n✅ EN！")