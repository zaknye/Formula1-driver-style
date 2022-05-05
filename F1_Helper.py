import numpy as np
from scipy import signal, stats
from sklearn.cluster import KMeans


# calculates the poeaks and valleys of speed data
# uses scipy peaks for maxes, and derivative local mins for valleys
def calc_min_max(data, threshold):
    grad = np.gradient(data)
    mins = np.array([i if i < threshold*-1 else 0 for i in grad])
    maxes = np.array([i if i > threshold else 0 for i in grad])
    mins = (np.diff(np.sign(mins)) > 0).nonzero()[0]
    maxes = (np.diff(np.sign(maxes)) < 0).nonzero()[0]
    
    mins = mins.flatten()
    maxes = maxes.flatten()
    maxes, _ = signal.find_peaks(data)
    return mins, maxes

# Calculates a dictionary containing the driver name, and the local min/max
# for each lap
def create_min_max_dict(driver, session):
    max_speed_list = np.empty((1,0))
    max_dist_list = np.empty((1,0))
    min_speed_list = np.empty((1,0))
    min_dist_list = np.empty((1,0))
    max_count = []
    min_count = []

    laps = session.laps.pick_driver(driver)
    for idx, lap in laps.iterlaps():
        # Check to make sure we only use laps that are representative of ultimate pace
        if lap.IsAccurate:
            tel = lap.get_car_data().add_distance()
            speed_np = np.array(tel['Speed'])
            mins, maxes = calc_min_max(speed_np, 0.7)
            max_select = np.array(tel['Speed'].iloc[maxes])
            min_select = np.array(tel['Speed'].iloc[mins])
            max_dist = np.array(tel['Distance'].iloc[maxes])
            min_dist = np.array(tel['Distance'].iloc[mins])
            max_speed_list = np.append(max_speed_list, max_select)
            max_dist_list = np.append(max_dist_list, max_dist)
            min_speed_list = np.append(min_speed_list, min_select)
            min_dist_list = np.append(min_dist_list, min_dist)
            max_count.append(len(maxes))
            min_count.append(len(mins))

    final_dict = {}
    final_dict['driver'] = driver
    final_dict['max'] = np.stack((max_dist_list, max_speed_list), axis=-1)
    final_dict['min'] = np.stack((min_dist_list, min_speed_list), axis=-1)
    final_dict['max_count'] = max_count
    final_dict['min_count'] = min_count

    return final_dict

def calc_kmeans(min_max_dict, n_max_cluster, n_min_cluster):
    kMeans_dict = {}
    kMeans_dict['driver'] = min_max_dict['driver']
    kMeans_dict['max'] = KMeans(n_clusters=n_max_cluster).fit(min_max_dict['max'])
    kMeans_dict['min'] = KMeans(n_clusters=n_min_cluster).fit(min_max_dict['min'])

    return kMeans_dict

def create_all_min_max_dict(session):
    all_driver_dict = {}
    for driver in session.drivers:
        all_driver_dict[driver] = create_min_max_dict(driver, session)
    
    # Remove driver data from drivers that didn't complete enough laps
    finished_laps = len(sorted(all_driver_dict.items(), key = lambda x: x[1]['max_count']))
    print(f'Number of Finished laps: {finished_laps}')
    remove_list = []
    for k, v in all_driver_dict.items():
        if len(v['max_count']) < finished_laps * 0.5:
            print(f"Oops! {k} didn't finish the race")
            remove_list.append(k)
    # Remove everyone who didn't make it around half distance
    for item in remove_list:
        del all_driver_dict[item]

    # Yes, we iterate over lists a lot but we need to clean the data before we start making assumptions
    all_driver_dict['average'] = {'min': np.empty((0,2)), 'max': np.empty((0,2)), 'min_count': [], 'max_count': []}
    all_driver_dict['average']['driver'] = 'average'
    for driver in session.drivers:
        # Check if the current driver is in our dictionary
        if driver in all_driver_dict.keys():
            all_driver_dict['average']['min'] = np.append(all_driver_dict['average']['min'], all_driver_dict[driver]['min'], axis=0)
            all_driver_dict['average']['max'] = np.append(all_driver_dict['average']['max'], all_driver_dict[driver]['max'], axis=0)
            all_driver_dict['average']['min_count'].append(all_driver_dict[driver]['min_count'])
            all_driver_dict['average']['max_count'].append(all_driver_dict[driver]['max_count'])

    # Flatten the count lists
    all_driver_dict['average']['min_count'] = [counts for sublist in all_driver_dict['average']['min_count'] for counts in sublist]
    all_driver_dict['average']['max_count'] = [counts for sublist in all_driver_dict['average']['max_count'] for counts in sublist]
    
    return all_driver_dict

# Creates a full dict of all kMeans objects for each driver as well as an average
# Also calculates the number of clusters required for the kMeans calculation
def calc_all_kmeans(all_driver_dict):
    n_max_cluster, _ = stats.mode(all_driver_dict['average']['max_count'])
    n_min_cluster, _ = stats.mode(all_driver_dict['average']['min_count'])
    all_kmeans_dict = {}
    for k, v in all_driver_dict.items():
        all_kmeans_dict[k] = calc_kmeans(v, n_max_cluster[0], n_min_cluster[0])

    return all_kmeans_dict

# creates a dictionary that stores all of the distances between the average points and driver specific points
# only uses the linear distance between the distance measurements, and not euclidean between distance and speed
def calc_distance_distance(all_kmeans_dict):
    distance_dict = {}
    avg_max_cluster_centers = all_kmeans_dict['average']['max'].cluster_centers_.transpose()[1]
    avg_min_cluster_centers = all_kmeans_dict['average']['min'].cluster_centers_.transpose()[1]
    for k, v in all_kmeans_dict.items():
        if not k == 'average':
            distance_dict[k] = {}
            distance_dict[k]['max'] = avg_max_cluster_centers - all_kmeans_dict[k]['max'].cluster_centers_.transpose()[1]
            distance_dict[k]['min'] = avg_min_cluster_centers - all_kmeans_dict[k]['min'].cluster_centers_.transpose()[1]
    
    return distance_dict