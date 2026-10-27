
CITY_COORDS = {
    "Delhi": (28.5562, 77.1000), "Mumbai": (19.0896, 72.8656),
    "Bengaluru": (13.1986, 77.7066), "Hyderabad": (17.2403, 78.4294),
    "Chennai": (12.9941, 80.1709), "Kolkata": (22.6547, 88.4467),
    "Jaipur": (26.8242, 75.8122), "Lucknow": (26.7606, 80.8893),
    "Ahmedabad": (23.0772, 72.6347), "Pune": (18.5822, 73.9197),
}

def haversine_km(source, destination):
    lat1, lon1 = CITY_COORDS[source]
    lat2, lon2 = CITY_COORDS[destination]
    radius = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return int(round(2 * radius * math.asin(math.sqrt(a))))

def build_features(airline, source, destination, journey_date, departure_time,
                   total_stops, travel_class, booking_days_before):
    travel_day = pd.to_datetime(journey_date)
    departure_hour = int(str(departure_time).split(":")[0])
    distance = haversine_km(source, destination)
    duration = int(round(42 + distance / 11.5 + int(total_stops) * 65))
    return pd.DataFrame([{
        "airline": airline, "source": source, "destination": destination,
        "travel_class": travel_class, "month": int(travel_day.month),
        "day_of_week": int(travel_day.dayofweek), "departure_hour": departure_hour,
        "total_stops": int(total_stops), "booking_days_before": int(booking_days_before),
        "distance_km": distance, "duration_mins": duration,
    }])[FEATURES]

example = build_features(
    "IndiGo", "Delhi", "Mumbai", "2026-12-15", "09:00", 0, "Economy", 21
)
example_prediction = float(model.predict(example)[0])
print(f"Example estimated fare: ₹{example_prediction:,.0f}")
display(example)